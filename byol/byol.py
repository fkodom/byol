import copy
from itertools import chain
from os import cpu_count
from typing import Tuple, Callable, Union, Dict, List

import torch
from torch import nn, optim, Tensor
import torch.nn.functional as f
from torch.utils.data import DataLoader

try:
    from pytorch_lightning import LightningModule as ByolModule
    lightning_imported = True
except ImportError:
    ByolModule = nn.Module
    lightning_imported = False

from byol.encoder import EncoderWrapper
from byol.utils import normalized_mse, mlp, default_aug


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
        self.augment = default_aug(image_size) if augment_fn is None else augment_fn
        self.beta = beta
        self.encoder = EncoderWrapper(
            model, projection_size, hidden_size, layer=hidden_layer
        )
        self.predictor = mlp(projection_size, projection_size, hidden_size)

        self.hparams = hparams
        self.train_dataset = hparams.get("train_dataset", None)
        self.val_dataset = hparams.get("val_dataset", None)
        self._target = None

        projection_size = hparams.get("projection_size", 256)
        num_classes = hparams.get("num_classes", 10)
        lr = hparams.get("lr", 1e-4)
        self.linear = nn.Linear(projection_size, num_classes)
        self.linear_optimizer = optim.Adam(self.linear.parameters(), lr=lr)

        self(torch.zeros(2, 3, *image_size))

    @property
    def target(self):
        if self._target is None:
            self._target = copy.deepcopy(self.encoder)
        return self._target

    def update_target(self):
        for p, pt in zip(self.encoder.parameters(), self.target.parameters()):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def predict(self, x: Tensor) -> Tensor:
        return self.predictor(self.encoder(x))

    def _byol_loss(self, x: Tensor, pred: Tensor) -> Tensor:
        with torch.no_grad():
            target = self.target.forward(x)
        return normalized_mse(pred, target)

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams.get("optimizer", "Adam"))
        weight_decay = self.hparams.get("weight_decay", 1e-6)
        param_dicts = [
            {
                "params": chain(self.encoder.parameters(), self.predictor.parameters()),
                "lr": self.hparams.get("lr", 1e-4),
            },
            {
                "params": self.linear.parameters(),
                "lr": self.hparams.get("linear_lr", 1e-3),
            },
        ]
        return optimizer(param_dicts, weight_decay=weight_decay)

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

    def training_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x = batch[0]
        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)

        pred1, pred2 = self.predict(x1), self.predict(x2)
        loss = torch.mean(self._byol_loss(x1, pred2) + self._byol_loss(x2, pred1))

        if self.hparams.get("train_classifier", False) and len(batch) > 1:
            y = batch[1]
            with torch.no_grad():
                encoded = self.forward(x.detach())
            pred = torch.log_softmax(self.linear(encoded), dim=-1)
            loss += f.nll_loss(pred, y)

        return {"loss": loss, "log": {"train_loss": loss}}

    @torch.no_grad()
    def validation_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x, y = batch
        x1, x2 = self.augment(x), self.augment(x)
        pred1, pred2 = self.predict(x1), self.predict(x2)
        loss = torch.mean(self._byol_loss(x1, pred2) + self._byol_loss(x2, pred1))

        if self.hparams.get("train_classifier", False) and len(batch) > 1:
            encoded = self.forward(x.detach())
            pred = torch.log_softmax(self.linear(encoded), dim=-1)
            loss += f.nll_loss(pred, y)

        metric_fns = self.hparams.get("metrics", ())
        metrics = {m.__name__: m(pred, y) for m in metric_fns}

        return {"loss": loss, "log": {"val_loss": loss, **metrics}}

    @torch.no_grad()
    def validation_epoch_end(self, outputs: List[Dict],) -> Dict:
        val_loss = sum(x["loss"] for x in outputs) / len(outputs)
        log = {
            k: sum(x["log"][k] for x in outputs) / len(outputs)
            for k in outputs[0]["log"].keys()
        }
        return {"val_loss": val_loss, "log": log}
