import torch

if torch.cuda.is_available():
    cache_dir = "/home/du/project/models"
else:
    cache_dir = "/Users/yule/.cache"
