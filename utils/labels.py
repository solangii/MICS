import torch
import numpy as np


def exponential_func(x, coef):
    a = torch.exp(coef * x)
    b = torch.exp(coef * torch.ones(1).cuda()) - 1
    return (a - 1) / b


def gaussian_func(x, h):
    x = 1 - x
    inner_exp = - x ** 2 / (2 * h ** 2)
    output = np.exp(inner_exp)
    return torch.tensor(output).cuda()


def adjusted_gaussian_func(x, h=0.2):
    bias = gaussian_func(0, h)
    scalar = 1 / (1 - bias)
    output = gaussian_func(x, h)
    output = scalar * (output - bias)
    return torch.tensor(output).cuda()


def sine_func(x):
    output = 0.5 * np.sin(2 * np.pi * x + np.pi / 2.) + 0.5 if x > 0.5 else 0
    return torch.tensor(output).cuda()


def piecewise_func(x, h1=0.5, h2=0.):
    slope = 1 / (1 - h1 - h2)
    y = slope * (x - h1)
    output = min(max(0, y), 1)
    return torch.tensor(output).cuda()
