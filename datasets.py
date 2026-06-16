import numpy as np
import torch
import os
import json

rng = np.random.default_rng(42)

def trop_poly(deg, dim, coeffs):
    raise NotImplementedError


def topksubset(k, dim=-1):
    return lambda x: torch.sum(torch.topk(x, k, dim=dim).values, dim=dim)
