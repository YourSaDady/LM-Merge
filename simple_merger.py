import argparse
import sys
import os
import shutil
import logging
import time
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
# REQUIRES sentencepiece, protobuf

import torch
import numpy as np
import random

# The following are already copied from the original MergeLM code:
## from model_merging_methods.merging_methods import MergingMethod
## from model_merging_methods.task_vector import TaskVector
## from utils.utils import get_param_names_to_merge, get_modules_to_merge
## from utils.read_retrieval_head import read_retrieval_head
## from model_merging_methods.mask_weights_utils import mask_model_weights, mask_input_with_mask
## from utils.test_model import test_model_completion, prepare_model_from_cpu, load_model_from_checkpoint


random.seed(13245)


# copy the functions from the original MergeLM code
def copy_params_to_model(params, model):  # copying code from model_merging_methods.merging_methods
    for param_name, param_value in model.named_parameters():
        if param_name in params:
            param_value.data.copy_(params[param_name])
        else:
            print(f"param_name {param_name} not in params")


def read_retrieval_head(retrieval_head_file="./Mistral-7B-v0.1.json", cutoff=0.1, *args, **kwargs):
    """
    retieval head file 格式: {"0-0": [0, ..., 0], "0-1": [0, ..., 0], ...}

    Read the retrieval heads from the file.
    The retrieval heads are sorted by their retrieval scores in descending order.
    :param retrieval_head_file: the file containing the retrieval heads
    :param cutoff: the cutoff score. The default behavior is that the function returns the heads with scores at least the cutoff.
    :param kwargs: exclude: a list of heads to exclude. These heads would not appear in the returned heads. The argument would be in the same format as the output of this function.
                      random: the number of heads to randomly sample. If provided, then cutoff is ignored. Heads will be randomly sampled. This is used as a baseline feature.
    :return: a list. Each element is a tuple of a head and its score. The head is a tuple of layer and head index. The score is the average score of the head.
    
    返回格式: [((layer_idx, head_idx), score), ...]
    """
    with open(retrieval_head_file, "r") as f:
        head_list = json.load(f)
    head_score_list = [([int(number) for number in item[0].split('-')], np.mean(item[1])) for item in head_list.items()]
    head_score_list = sorted(head_score_list, key=lambda x: x[1], reverse=True)
    if kwargs.get("exclude") is not None:
        exclude = set(tuple(x[0]) for x in kwargs.get("exclude"))
        head_score_list = [item for item in head_score_list if tuple(item[0]) not in exclude]
    if kwargs.get("random") is not None:
        head_score_list = random.sample(head_score_list, kwargs.get("random"))
    else:
        i = 0
        for i, (head, score) in enumerate(head_score_list):
            if score < cutoff:
                print(f"{i} of {len(head_score_list)} heads ({i / len(head_score_list)}) have score at least {cutoff}")
                return head_score_list[:i]

    return head_score_list


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict,
        tokenizer,
        model,
):
    """Resize tokenizer and embedding by adding special tokens and averaging the embeddings of the new tokens.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    assert tokenizer.vocab_size == 32000
    print("Original vocab size: 32000")
    print(tokenizer.special_tokens_map)
    print(special_tokens_dict)
    tokenizer.add_special_tokens({"additional_special_tokens": []})  # a bug in huggingface tokenizers
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    print(f"Added {num_new_tokens} new tokens to the tokenizer.")
    # print all the special tokens
    print(tokenizer.special_tokens_map)
    if num_new_tokens > 0:
        model.resize_token_embeddings(tokenizer.vocab_size + num_new_tokens)

        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def mask_input_with_mask(input_tensor: torch.Tensor, mask: torch.Tensor, use_rescale: bool, mask_rate=None): #实现parameter drop的function
    """
    mask the input with a given mask
    mask: same shape as input_tensor, 1 for the parameters that we want to keep, 0 for the parameters that we want to mask
    mask_rate: float. If none, calculate the mask rate as the ratio of the number of parameters that we want to keep to the total number of parameters
    """
    masked_input_tensor = input_tensor * mask
    if use_rescale:
        if mask_rate is None:
            mask_rate = torch.sum(mask) / torch.numel(mask) #该weight tensor里head所占据的params的比例 （很小）
            mask_rate = 1 - mask_rate.item() #至此，mask_rate表示丢掉的param的占比

        if mask_rate < 1.0:
            if mask_rate == 0: #no head params inside this weight tensor
                print("Warning: mask_rate is 0. Not rescaling.")
                return masked_input_tensor
            masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate) #other表示保留率，将input除以保留率实现rescale
    return masked_input_tensor


def test_model_completion(enc, model, prompt):  # test model output using prompt
    prompt = enc(prompt, return_tensors="pt")
    input_ids = prompt["input_ids"].to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=50)
        response = enc.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response


def diff(model1, model2):  # compare two models
    for name1, param1 in model1.named_parameters():
        param2 = model2.state_dict()[name1]
        if not torch.equal(param1, param2):
            print(name1, "different by", torch.norm(param1 - param2).item())


######################################
# Main code
######################################

###### arguments ######
merge_model1 = False
cutoff = 0.0
rescale = True
save_model = True
substitute_linear = False
substitute_io = False
resize_tokenizer = False  # DO NOT MODIFY THIS DURING DIFFERENT CALLS TO do_model_merge().
linear_rescale = False
linear_keep_rate = 0.5
# merging_method = "double_delta"  # "simple_substitution", "simple_delta", "double_delta", "double_sample", "double_sample_exclusive"
merging_method = "simple_delta"
linear_merging_method = "double_delta"  # "simple_substitution", "simple_delta", "double_delta", "double_sample", "model_vector"
double_sample_cnt = 120
mlp_block_keep_configs = {  # if the value is False, the corresponding block is replaced with the model2's block
    "gate_proj": True,
    "up_proj": True,
    "down_proj": True,
    "keep_retrieval_distribution": True
}
linear_rescale_fixed_rate = True

## hard-coded model features
num_key_value_groups = 4 #每组key value heads pair对应4个query heads (Mistral系列模型架构采用GQA)
head_dim = 128
num_layers = 32  # hard-coded


#######################

def base_name():
    if merge_model1:
        return "m"
    else:
        return "lco"


def get_merging_method():
    if merging_method in ["double_sample", "double_sample_exclusive"]:
        return merging_method + "_" + str(double_sample_cnt)
    else:
        return merging_method


model1_path = "../../models/Eurus-7b-sft"  # hard-coded
# model2_path = "../../models/MistralLite"  # hard-coded
model2_path = "../../models/Eurus-7b-sft"
base_model_path = "../../models/Mistral-7B-v0.1"  # hard-coded

base_enc = AutoTokenizer.from_pretrained(base_model_path)
enc1 = AutoTokenizer.from_pretrained(model1_path)
enc2 = AutoTokenizer.from_pretrained(model2_path)
if merge_model1:
    model1 = AutoModelForCausalLM.from_pretrained(model1_path)
model2 = AutoModelForCausalLM.from_pretrained(model2_path)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

if resize_tokenizer: #default=False
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="[PAD]",
                                 additional_special_tokens=['<unk>', '<s>', '</s>', "<|assistant|>", "<|prompter|>"]),
        model=base_model,
        tokenizer=base_enc,
    )
    if merge_model1:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]",
                                     additional_special_tokens=['<unk>', '<s>', '</s>', "<|assistant|>",
                                                                "<|prompter|>"]),
            model=model1,
            tokenizer=enc1,
        )

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="[PAD]",
                                 additional_special_tokens=['<unk>', '<s>', '</s>', "<|assistant|>", "<|prompter|>"]),
        model=model2,
        tokenizer=enc2,
    )

model_template_path = base_model_path  # or model2_path #?


# model_template_path = model2_path


def get_kv_and_q_indices(*retrieval_head_args, **retrieval_head_kwargs): #? 返回list of retrieval heads, 以及两个dictionary分别记录mask掉k-vpairs以及q的head indicies
    head_indices_to_mask_kv = {}
    head_indices_to_mask_q = {}
    retrieval_heads = read_retrieval_head(*retrieval_head_args, **retrieval_head_kwargs)
    keep_ratio = len(retrieval_heads) / 1024  ## hard_coded (默认总共有1024个attention heads)
    print("Keep ratio: ", keep_ratio)

    for retrieval_head, _ in retrieval_heads:
        layer, head = retrieval_head #都是idx
        if layer not in head_indices_to_mask_kv:
            head_indices_to_mask_kv[layer] = []
            head_indices_to_mask_q[layer] = []
        l1, l2 = head_indices_to_mask_kv[layer], head_indices_to_mask_q[layer]
        if head not in l2: #q heads直接放到每个layer的列表里
            l2.append(head)
        if head // num_key_value_groups not in l1: #k heads 和 v heads的总共列表长度是q heads列表除以kv group size后的长度（因为每个k head, v headd对应4个q heads)
            l1.append(head // num_key_value_groups)
    return retrieval_heads, head_indices_to_mask_kv, head_indices_to_mask_q


def do_model_merge():
    global substitute_io #FALSE
    #这里都是base model的
    model_to_merge = AutoModelForCausalLM.from_pretrained(model_template_path) 
    enc_to_merge = AutoTokenizer.from_pretrained(model_template_path)

    # it is important to use the config of the model that we want to merge
    model_to_merge.config = model2.config #Eurus-7B

    if resize_tokenizer: #false
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]",
                                     additional_special_tokens=['<unk>', '<s>', '</s>', "<|assistant|>",
                                                                "<|prompter|>"]),
            model=model_to_merge,
            tokenizer=enc_to_merge,
        )

    else:  # if we do not resize the tokenizer, we should keep the io layer (subsitute_io should be False)
        if substitute_io:
            print(
                "Warning: tokenizer is not resized, but substitute_io is True. This is not feasible. Changing substitute_io to False.")
            substitute_io = False

    model_to_merge.config.vocab_size = enc_to_merge.vocab_size

    enc_to_merge = base_enc

    # merger = MergingMethod("retrieval_head_merging")

    if merge_model1:
        model1_param_dict = dict(model1.named_parameters())
    model2_param_dict = dict(model2.named_parameters())
    base_model_param_dict = dict(base_model.named_parameters())

    # Get the head indices to keep from retrieval heads.
    # These dicts are used to mask the parameters of the model. The keys are the layer indices. The values are lists of head indices to keep in the corresponding layers.
    # Since Mistral uses GQA, we deal with kv and q projections separately. (output_proj is also done int the same way as q_proj)

    retrieval_heads, head_indices_to_mask_kv, head_indices_to_mask_q = get_kv_and_q_indices(cutoff=cutoff) #cutoff=0

    retrieval_head_distribution = [0] * num_layers #32
    for layer, heads in head_indices_to_mask_q.items():
        retrieval_head_distribution[layer] = len(heads)
    retrieval_head_distribution = np.array(retrieval_head_distribution) / np.sum(retrieval_head_distribution)
    print("Retrieval head distribution: ", retrieval_head_distribution) #32layers, 每个layer显示mask q的heads数量(默认是所有attention heads)

    #####################

    if merge_model1:
        if merging_method in ["double_sample", "double_sample_exclusive"]:
            if merging_method == "double_sample": #random sample for 120 heads
                retrieval_head_kwargs = {"random": double_sample_cnt}
            else:
                retrieval_head_kwargs = {"random": double_sample_cnt, "exclude": retrieval_heads}
        else:
            retrieval_head_kwargs = {"cutoff": 0.00}

        heads_to_keep_model1, head_indices_to_mask_kv1, head_indices_to_mask_q1 = get_kv_and_q_indices(
            **retrieval_head_kwargs)

    #####################

    masks = []

    merged_params = {}

    with torch.no_grad():
        for param_name, param_value in base_model_param_dict.items(): # 一个element是对应一个weight  matrix吗?
            print("param_name: ", param_name)
            if 'self_attn' in param_name and "_proj" in param_name:  # Dealing with attention heads
                print("If")
                layer_idx = int(param_name.split(".")[2])
                mask = torch.zeros_like(param_value) #这个mask最终会mask掉k/q/v heads对应的params(标记为1)跳过drop 环节
                mask_model1 = torch.zeros_like(param_value)
                masks.append((mask, mask_model1))
                if layer_idx in head_indices_to_mask_kv: #如果该层包含k heads或者v heads（一般都有）, 要mask掉k/v/q heads所在的param位置？？（相当merge的时候保留retrieval heads的 weights跳过drop环节）
                    if "k_proj" in param_name or "v_proj" in param_name:
                        indices = head_indices_to_mask_kv[layer_idx]
                    else:
                        indices = head_indices_to_mask_q[layer_idx] #为什么会有这个else?
                    # print("indices", indices)
                    for index in indices: 
                        mask[head_dim * index: head_dim * (index + 1)] = 1 #head_dim=128

                if merge_model1:
                    if layer_idx in head_indices_to_mask_kv1:
                        if "k_proj" in param_name or "v_proj" in param_name:
                            indices = head_indices_to_mask_kv1[layer_idx]
                        else:
                            indices = head_indices_to_mask_q1[layer_idx]
                        # print("indices", indices)
                        for index in indices:
                            mask_model1[head_dim * index: head_dim * (index + 1)] = 1
                mask_rev = 1 - mask
                if merging_method == "simple_substitution":
                    raise NotImplementedError("simple_substitution not implemented")
                    use_delta = use_delta1 = False
                elif merging_method == "simple_delta":
                    use_delta = True
                    use_delta1 = False
                elif merging_method == "double_delta": #default
                    use_delta = use_delta1 = True
                elif merging_method in ["double_sample", "double_sample_exclusive"]:
                    use_delta = use_delta1 = True
                else:
                    raise ValueError(f"merging method {merging_method} not recognized")

                if use_delta:
                    delta = model2_param_dict[param_name] - base_model_param_dict[param_name]
                    masked_delta = mask_input_with_mask(delta, mask, use_rescale=rescale, mask_rate=None) #实现drop以及rescale
                    if merge_model1:
                        # drop and scale
                        if use_delta1:
                            delta1 = model1_param_dict[param_name] - base_model_param_dict[param_name]
                            if merging_method in ["double_sample", "double_sample_exclusive"]:
                                mask1 = mask_model1
                            else:
                                mask1 = mask_rev
                            masked_delta1 = mask_input_with_mask(delta1, mask1, use_rescale=False, mask_rate=None)
                            masked_tensor1 = base_model_param_dict[param_name] + masked_delta1 + masked_delta #double delta就是把两个要被融合的models的 delta都加上
                            masked_tensor2 = None
                        else:
                            masked_tensor1 = mask_input_with_mask(model1_param_dict[param_name], mask_rev,
                                                                  use_rescale=False, mask_rate=None)
                            masked_tensor2 = mask_input_with_mask(masked_delta + base_model_param_dict[param_name],
                                                                  mask, use_rescale=False, mask_rate=None)
                    else:
                        # simple delta only
                        if merging_method != "simple_delta":
                            print(
                                "Warning: when merge_model1 is False, the only supported merging_method is simple_delta. Changing merging_method to simple_delta.")
                        masked_tensor1 = masked_delta
                        masked_tensor2 = base_model_param_dict[param_name]
                else:
                    # mask1 is the inverse of mask; this is the default technique
                    masked_tensor1 = mask_input_with_mask(base_model_param_dict[param_name], mask_rev,
                                                          use_rescale=False, mask_rate=None)
                    # masked_tensor1 = mask_input_with_mask(model1_param_dict[param_name], mask1, use_rescale=False, mask_rate=None)
                    masked_tensor2 = mask_input_with_mask(model2_param_dict[param_name], mask, use_rescale=False,
                                                          mask_rate=None)
                merged_params[param_name] = (masked_tensor1, masked_tensor2)  # tensor1是drop以及rescale之后的delta weights, tensor2是base model的weight tensor
            else:  # Dealing with other parameters
                print("Else")
                substitute_flag = False
                keep_rate = 1
                if param_name.find("embed_tokens") != -1 or param_name.find("lm_head") != -1:
                    if substitute_io is True:
                        substitute_flag = True
                        # keep_rate = 1
                    else: #default
                        substitute_flag = False
                else:
                    if substitute_linear is True:
                        for name in mlp_block_keep_configs:
                            if param_name.find(name) != -1:
                                if mlp_block_keep_configs[name] is False:
                                    substitute_flag = True
                                    keep_rate = linear_keep_rate
                                    break
                    else: #default
                        substitute_flag = False

                if substitute_flag:
                    if keep_rate == 1:
                        merged_params[param_name] = (model2_param_dict[param_name], None)  # only model2 is used
                    else:
                        delta = model2_param_dict[param_name] - base_model_param_dict[param_name]
                        if mlp_block_keep_configs["keep_retrieval_distribution"]: #?
                            layer_idx = int(param_name.split(".")[2])
                            # if retrieval_head_distribution[layer_idx] == 0:
                            #     print("Warning: retrieval_head_distribution is 0 for layer_idx: ", layer_idx)
                            #     floating_keep_rate = 0
                            # else:
                            #     floating_keep_rate = keep_rate
                            floating_keep_rate = keep_rate * retrieval_head_distribution[layer_idx] * len(
                                retrieval_head_distribution)
                        else:
                            floating_keep_rate = keep_rate
                        print("Floating keep rate: ", floating_keep_rate, "for param_name: ", param_name, "layer_idx: ",
                              layer_idx)

                        if floating_keep_rate == 0:
                            print("Warning: floating_keep_rate is 0. Not rescaling.")
                            masked_delta = torch.zeros_like(delta)
                        else:
                            mask = torch.rand_like(delta) < floating_keep_rate
                            if linear_rescale_fixed_rate:
                                masked_delta = mask_input_with_mask(delta, mask, use_rescale=linear_rescale,
                                                                    mask_rate=keep_rate)
                            else: #drop rate not fixed?
                                masked_delta = mask_input_with_mask(delta, mask, use_rescale=linear_rescale,
                                                                    mask_rate=None)

                        if merge_model1:
                            delta1 = model1_param_dict[param_name] - base_model_param_dict[param_name]

                            if linear_merging_method == "simple_substitution":
                                raise NotImplementedError("simple_substitution not implemented")
                                masked_delta1 = mask_input_with_mask(delta1, ~mask, use_rescale=False, mask_rate=None)
                                masked_tensor1 = base_model_param_dict[param_name] + masked_delta1 + masked_delta
                                merged_params[param_name] = (masked_tensor1, None)
                            elif linear_merging_method == "simple_delta":
                                masked_delta1 = mask_input_with_mask(delta1, ~mask, use_rescale=False, mask_rate=None)
                                masked_tensor1 = base_model_param_dict[param_name] + masked_delta1 + masked_delta
                                merged_params[param_name] = (masked_tensor1, None)
                            elif linear_merging_method == "double_delta" or linear_merging_method == "double_sample":
                                if linear_merging_method == "double_delta":
                                    mask1 = ~mask
                                else:
                                    mask1 = torch.rand_like(delta1) < floating_keep_rate
                                masked_delta1 = mask_input_with_mask(delta1, mask1, use_rescale=True, mask_rate=None)
                                masked_tensor1 = base_model_param_dict[param_name] + masked_delta1 + masked_delta
                                merged_params[param_name] = (masked_tensor1, None)
                            elif linear_merging_method == "model_vector":
                                raise NotImplementedError("model_vector not implemented")

                        else:
                            if linear_merging_method != "simple_delta":
                                print("Warning: when merge_model1 is False, the only supported linear_merging_method "
                                      "is simple_delta. Changing linear_merging_method to simple_delta.")
                            masked_tensor = base_model_param_dict[param_name] + masked_delta
                            merged_params[param_name] = (masked_tensor, None)
                else:
                    if merge_model1:
                        merged_params[param_name] = (model1_param_dict[param_name], None)
                    else:
                        merged_params[param_name] = (base_model_param_dict[param_name], None) #tensor1是base model的weight tensor
                # merged_params[param_name] = (param_value, None)  # only model1 is used
    # 这里是最后一步处理merged models的params
    merged_params1 = {}
    for k, (v1, v2) in merged_params.items():
        if v2 is None: #non-head
            merged_params1[k] = v1
        else: #head
            merged_params1[k] = v1 + v2

    copy_params_to_model(params=merged_params1, model=model_to_merge)

    if save_model:
        save_path = (f"../../models/{base_name()}_{cutoff}_keep_{len(retrieval_heads)}_" #../models/lco_0.03_keep_1024_?
                     f"rescale_{rescale}_substitute_io_{substitute_io}_substitute_linear_{substitute_linear}_"
                     f"resized_{resize_tokenizer}_lkr_{linear_keep_rate}_linear_rescale_{linear_rescale}_"
                     f"{get_merging_method()}_linear_merging_method_{linear_merging_method}_mlpk_"
                     f"{mlp_block_keep_configs['gate_proj']}_{mlp_block_keep_configs['up_proj']}_"
                     f"{mlp_block_keep_configs['down_proj']}_"
                     f"krdf_{mlp_block_keep_configs['keep_retrieval_distribution']}")
        model_to_merge.save_pretrained(save_path)
        enc_to_merge.save_pretrained(save_path)
        print("Model saved to: ", save_path)


# diff(model_to_merge, model2)

# model_to_merge=model_to_merge.to("cuda")
# print(test_model_completion(enc_to_merge, model_to_merge, "The capital of the United States is") )
# print(test_model_completion(enc_to_merge, model_to_merge, "[User] Write a python code that solves the problem: given a list of numbers, find the maximum and minimum elements. Your code should be a function that takes a list as the only argument, and return a tuple, which consists of four numbers: the maximum value, the index where the first maximum value appears, the minimum value, and the index where the first minimum value appears. [Assistant]"))
# del model_to_merge


# base_model=base_model.to("cuda:1")
# print(test_model_completion(base_enc, base_model, "[User] Write a python code that solves the problem: given a list of numbers, find the maximum and minimum elements. Your code should be a function that takes a list as the only argument, and return a tuple, which consists of four numbers: the maximum value, the index where the first maximum value appears, the minimum value, and the index where the first minimum value appears. [Assistant]"))
#
# model1=model1.to("cuda:2")
# print(test_model_completion(enc1, model1, "[User] Write a python code that solves the problem: given a list of numbers, find the maximum and minimum elements. Your code should be a function that takes a list as the only argument, and return a tuple, which consists of four numbers: the maximum value, the index where the first maximum value appears, the minimum value, and the index where the first minimum value appears. [Assistant]"))
# model2=model2.to("cuda:3")
# print(test_model_completion(enc2, model2, "[User] Write a python code that solves the problem: given a list of numbers, find the maximum and minimum elements. Your code should be a function that takes a list as the only argument, and return a tuple, which consists of four numbers: the maximum value, the index where the first maximum value appears, the minimum value, and the index where the first minimum value appears. [Assistant]"))
#


cutoffs = [0.03]
substitute_ios = [False]
substitute_linears = [True]
linear_keep_rates = [0.1]
linear_rescales = [True]
merging_methods = ["double_delta", "simple_delta", "double_sample", "double_sample_exclusive"][1:2] #simple_delta
linear_merging_methods = ["double_delta", "simple_delta", "double_sample"][1:2] #simple_delta
double_sample_cnts = [120]
mlp_block_keep_configs_gates = [True, False][1:2] #false
mlp_block_keep_configs_up = [True, False][1:2] #false
mlp_block_keep_configs_down = [True, False][1:2] #false
mlp_block_keep_configs_retrievals = [True]

for cutoff in cutoffs:
    for substitute_io in substitute_ios:
        for substitute_linear in substitute_linears:
            for linear_keep_rate in linear_keep_rates:
                for linear_rescale in linear_rescales:
                    for merging_method in merging_methods:
                        for linear_merging_method in linear_merging_methods:
                            for double_sample_cnt in double_sample_cnts:
                                for mlp_block_keep_config_gate in mlp_block_keep_configs_gates:
                                    for mlp_block_keep_config_up in mlp_block_keep_configs_up:
                                        for mlp_block_keep_config_down in mlp_block_keep_configs_down:
                                            for mlp_block_keep_config_retrieval in mlp_block_keep_configs_retrievals:
                                                mlp_block_keep_configs = {
                                                    "gate_proj": mlp_block_keep_config_gate,
                                                    "up_proj": mlp_block_keep_config_up,
                                                    "down_proj": mlp_block_keep_config_down,
                                                    "keep_retrieval_distribution": mlp_block_keep_config_retrieval
                                                }
                                                print("=====================================")
                                                print("Starting with cutoff: ", cutoff)
                                                print("substitute_io: ", substitute_io)
                                                print("substitute_linear: ", substitute_linear)
                                                print("linear_keep_rate: ", linear_keep_rate)
                                                print("linear_rescale: ", linear_rescale)
                                                print("merging_method: ", merging_method)
                                                print("linear_merging_method: ", linear_merging_method)
                                                print("double_sample_cnt: ", double_sample_cnt)
                                                print("mlp_block_keep_configs: ", mlp_block_keep_configs)

                                                print("=====================================")
                                                do_model_merge()
                                                print(
                                                    f"Done with {cutoff}_{substitute_io}_{substitute_linear}_{linear_keep_rate}_{linear_rescale}_{merging_method}_{linear_merging_method}_{double_sample_cnt}_{mlp_block_keep_configs}")
                                                print("=====================================")
