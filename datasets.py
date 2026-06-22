import numpy as np
import torch
import os
import json

rng = np.random.default_rng(42)

def trop_poly(deg, dim, coeffs):
    raise NotImplementedError


def topksubset(k, dim=-1):
    """
    A function that takes a tensor x and returns the sum of the top-k elements along the specified dimension.

    Args:
        k: number of top elements to sum
        dim: dimension along which to compute the top-k sum (default: -1)
    """
    return lambda x: torch.sum(torch.topk(x, k, dim=dim).values, dim=dim)

def x2(dim):
    """
    A function that takes a tensor x and returns the sum of the squares of its elements.
    Args:
        dim: dimension along which to compute the top-k sum (default: -1)
    """
    return lambda x: torch.sum(x**2, dim=dim)

def sin():
    return torch.sin
