import copy
import random
from typing import Tuple, Callable, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as f

from kornia import augmentation as aug
from kornia import filters


class RandomApply(nn.Module):
    def __init__(self, fn: Callable, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x if random.random() > self.p else self.fn(x)


def normalized_mse(x: Tensor, y: Tensor) -> Tensor:
    return 2 - 2 * (f.normalize(x, dim=-1) * f.normalize(y, dim=-1)).sum(dim=-1)


def mlp(dim: int, projection_size: int = 256, hidden_size: int = 4096) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size),
    )


class ModelWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        projection_size: int = 256,
        hidden_size: int = 4096,
        layer: Union[str, int] = -2,
    ):
        super().__init__()
        self.model = model
        self.projection_size = projection_size
        self.hidden_size = hidden_size
        self.layer = layer

        self._projector = None
        self.hidden = None
        self._register_hook()

    @property
    def projector(self):
        if self._projector is None:
            _, dim = self.hidden.shape
            self._projector = mlp(dim, self.projection_size, self.hidden_size)
        return self._projector

    def _hook(self, _, __, output):
        self.hidden = output.flatten(start_dim=1)

    def _register_hook(self):
        if isinstance(self.layer, str):
            layer = dict([*self.model.named_modules()])[self.layer]
        else:
            layer = list(self.model.children())[self.layer]

        layer.register_forward_hook(self._hook)

    def forward(self, x: Tensor) -> Tensor:
        _ = self.model(x)
        return self.projector(self.hidden)


class BYOL(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        image_size: Tuple[int, int] = (256, 256),
        hidden_layer: Union[str, int] = -2,
        projection_size: int = 256,
        hidden_size: int = 4096,
        augment_fn: Callable = None,
        beta: float = 0.99,
    ):
        super().__init__()
        DEFAULT_AUG: Callable = nn.Sequential(
            RandomApply(aug.ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8),
            aug.RandomGrayscale(p=0.2),
            aug.RandomHorizontalFlip(),
            RandomApply(filters.GaussianBlur2d((3, 3), (1.5, 1.5)), p=0.1),
            aug.RandomResizedCrop(image_size),
            aug.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            ),
        )
        self.augment: Callable = DEFAULT_AUG if augment_fn is None else augment_fn
        self.encoder = ModelWrapper(
            model, projection_size, hidden_size, layer=hidden_layer
        )
        self.target = copy.deepcopy(self.encoder)
        self.beta = beta
        self.predictor = mlp(projection_size, projection_size, hidden_size)

        device = list(self.parameters())[0].device
        self(torch.zeros(2, 3, *image_size, device=device))

    def update_target(self):
        for p, pt in zip(self.encoder.parameters(), self.target.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)

        y1 = self.predictor(self.encoder(x1))
        y2 = self.predictor(self.encoder(x2))
        with torch.no_grad():
            target1 = self.target(x1)
            target2 = self.target(x2)

        return torch.mean(normalized_mse(y1, target2) + normalized_mse(y2, target1))
