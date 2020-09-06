import random
from typing import Callable, Tuple

from kornia import augmentation as aug
from kornia import filters
from kornia.geometry import transform as tf
import torch
from torch import nn, Tensor
import torch.nn.functional as f


class RandomApply(nn.Module):
    def __init__(self, fn: Callable, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x if random.random() > self.p else self.fn(x)


DEFAULT_AUG: Callable = nn.Sequential(
    tf.Resize(size=(256, 256)),
    RandomApply(aug.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
    aug.RandomGrayscale(p=0.2),
    aug.RandomHorizontalFlip(),
    RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
    aug.RandomResizedCrop(size=(256, 256)),
    aug.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225]),
    ),
)


def normalized_mse(x: Tensor, y: Tensor) -> Tensor:
    x = f.normalize(x, dim=-1)
    y = f.normalize(y, dim=-1)
    return 2 - 2 * (x * y).sum(dim=-1)


def mlp(dim: int, projection_size: int = 256, hidden_size: int = 4096) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size),
    )
