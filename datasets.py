import numpy as np
import torch
import os
import json

rng = np.random.default_rng(42)

def topksubset(X, k):
    torch.sum(torch.topk(X, k, dim=-1).values, dim=-1)
