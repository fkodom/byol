from os import cpu_count
from typing import Dict, List, Union, Tuple, Callable

import torch
from torch import nn, optim, Tensor
import torch.nn.functional as f
from torch.utils.data import DataLoader
try:
    from pytorch_lightning import LightningModule
except ImportError:
    LightningModule = nn.Module

from byol import BYOL


class ByolLightningModule(LightningModule):
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
        self.byol = BYOL(
            model=model,
            image_size=image_size,
            hidden_layer=hidden_layer,
            projection_size=projection_size,
            hidden_size=hidden_size,
            augment_fn=augment_fn,
            beta=beta,
        )
        self.hparams = hparams
        self.train_dataset = hparams.get("train_dataset", None)
        self.val_dataset = hparams.get("val_dataset", None)

        projection_size = hparams.get("projection_size", 256)
        num_classes = hparams.get("num_classes", 10)
        lr = hparams.get("lr", 1e-4)
        self.linear = nn.Linear(projection_size, num_classes)
        self.linear_optimizer = optim.Adam(self.linear.parameters(), lr=lr)

    @property
    def encoder(self):
        return self.byol.encoder

    def forward(self, x: Tensor) -> Tensor:
        return self.byol.forward(x)

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
        loss.backward()

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
