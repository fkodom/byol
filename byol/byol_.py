import copy
from os import cpu_count
import random
from typing import Tuple, Callable, Union, Dict, List

import torch
from torch import nn, optim, Tensor
import torch.nn.functional as f
from torch.utils.data import DataLoader
try:
    from pytorch_lightning import LightningModule as ByolModule
except ImportError:
    ByolModule = nn.Module

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


class BYOL(ByolModule):
    def __init__(
        self,
        model: nn.Module,
        image_size: Tuple[int, int] = (256, 256),
        hidden_layer: Union[str, int] = -2,
        projection_size: int = 256,
        hidden_size: int = 4096,
        augment_fn: Callable = None,
        beta: float = 0.99,
        **hparams,
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

        self.hparams = hparams
        self.train_dataset = hparams.get("train_dataset", None)
        self.val_dataset = hparams.get("val_dataset", None)

        projection_size = hparams.get("projection_size", 256)
        num_classes = hparams.get("num_classes", 10)
        lr = hparams.get("lr", 1e-4)
        self.linear = nn.Linear(projection_size, num_classes)
        self.linear_optimizer = optim.Adam(self.linear.parameters(), lr=lr)

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

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams.get("optimizer", optim.Adam))
        lr = self.hparams.get("lr", 1e-4)
        weight_decay = self.hparams.get("weight_decay", 1e-6)
        return optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

    def _dataloader(self, train: bool) -> DataLoader:
        batch_size = self.hparams.get("batch_size", 128)
        num_workers = self.hparams.get("num_workers", cpu_count())

        return DataLoader(
            self.train_dataset if train else self.val_dataset,
            batch_size=batch_size,
            shuffle=train,
            drop_last=train,
            num_workers=num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(train=False)

    def training_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        x, y = batch
        loss = self.forward(x)
        loss.backward(retain_graph=True)

        with torch.no_grad():
            encoded = self.encoder.projector(self.encoder.hidden)

        self.linear_optimizer.zero_grad()
        pred = torch.log_softmax(self.linear(encoded), dim=-1)
        linear_loss = f.nll_loss(pred, y)
        linear_loss.backward()
        self.linear_optimizer.step()

        return {
            "loss": loss,
            "log": {"train_loss": loss, "train_linear_loss": linear_loss},
        }

    @torch.no_grad()
    def validation_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        x, y = batch
        loss = self.forward(x)
        encoded = self.encoder.projector(self.encoder.hidden)
        pred = torch.log_softmax(self.linear(encoded), dim=-1)
        linear_loss = f.nll_loss(pred, y)

        metric_fns = self.hparams.get("metrics", ())
        metrics = {m.__name__: m(pred, y) for m in metric_fns}

        return {
            "loss": loss,
            "log": {"val_loss": loss, "val_linear_loss": linear_loss, **metrics},
        }

    @torch.no_grad()
    def validation_epoch_end(
        self, outputs: List[Dict[str, Tensor]],
    ) -> Dict[str, Union[Tensor, Dict[str, Tensor]]]:
        metrics = self.hparams.get("metrics", ())
        val_loss = sum(x["loss"] for x in outputs) / len(outputs)
        log = {
            m.__name__: sum(x[m.__name__] for x in outputs) / len(outputs)
            for m in metrics
        }

        return {"val_loss": val_loss, "log": log}
