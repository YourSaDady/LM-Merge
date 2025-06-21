import torch
import tensor_parallel as tp
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_model_completion(enc, model, prompt):
    prompt = enc(prompt, return_tensors="pt")
    input_ids = prompt["input_ids"].to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=50)
        response = enc.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response


def prepare_model_from_cpu(model):
    model = tp.tensor_parallel(model, sharded=True)
    return model

def load_model_from_checkpoint(model_path):
    enc = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return enc, model