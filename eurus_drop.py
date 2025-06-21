# TODO:
# 1. Load Eurus-7b-sft from https://huggingface.co/openbmb/Eurus-7b-sft
# 2. Load Mistral-7B-v0.1 from mistralai/Mistral-7B-v0.1
# 3. Randomly keep only p (0<=p<=1) of the parameters of Eurus-7b-sft. The way to do this is:
#    - Calculate delta = (Eurus-7b-sft - Mistral-7B-v0.1)
#    - Apply a mask to delta, where the mask is 1 with probability p and 0 with probability 1-p
#    - Rescale delta by 1/p (or not)
#    - Add delta to Mistral-7B-v0.1
# 4. Save the new model as Eurus-7b-sft-drop-p
# 5. For different values of p, do zero-shot evaluation on GSM8K. Plot the results.
# Research Questions:
# 1. How does the performance of the model change with the percentage of parameters dropped? How many parameters can be dropped before the performance drops significantly?
# 2. Does rescaling delta by 1/p help in maintaining the performance of the model?
# 3. Change the drop policy. Instead of randomly dropping the parameters, drop the parameters with the highest retrieval scores. The reason we do this is that we want to keep retrieval heads of the long context model when merging. How does this change the performance of the model?

'''
1. Load Eurus-7b-sft
'''
