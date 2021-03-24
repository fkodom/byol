from itertools import chain
from typing import Tuple, Callable, Union, Dict, List

import torch
from torch import nn, optim, Tensor
import torch.nn.functional as f

from src.extractor import FeatureExtractor
from src.utils import normalized_mse, default_augmentation, mlp

try:
    from pytorch_lightning import LightningModule
except ImportError:
    LightningModule = nn.Module



class BYOL(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        image_size: Tuple[int, int] = (128, 128),
        hidden_layer: Union[str, int] = -2,
        projection_size: int = 256,
        hidden_size: int = 4096,
        augment_fn: Callable = None,
        beta: float = 0.99,
        **hparams,
    ):
        super().__init__()
        self.augment = augment_fn if augment_fn else default_augmentation(image_size)
        self.beta = beta
        self.hparams = hparams
        self.extractor = FeatureExtractor(model, layer=hidden_layer)

        dummy_inputs = torch.zeros(2, 3, *image_size)
        extracted_size = self.extractor(dummy_inputs).size(-1)
        num_classes = hparams.get("num_classes", 10)
        self.projector = mlp(extracted_size, projection_size, hidden_size)
        self.predictor = mlp(projection_size, projection_size, hidden_size)
        self.linear = nn.Linear(extracted_size, num_classes)

        self.target_extractor = FeatureExtractor(model, layer=hidden_layer)
        self.target_projector = mlp(extracted_size, projection_size, hidden_size)
        self.target_projector.load_state_dict(self.projector.state_dict())

    def update_targets(self):
        for p, pt in zip(
            self.extractor.parameters(), self.target_extractor.parameters()
        ):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data
        for p, pt in zip(
            self.projector.parameters(), self.target_projector.parameters()
        ):
            pt.data = self.beta * pt.data + (1 - self.beta) * p.data

    def forward(self, x: Tensor) -> Tensor:
        return self.extractor(x).flatten(start_dim=1)

    def project(self, x: Tensor) -> Tensor:
        return self.projector(self.forward(x))

    def predict(self, x: Tensor) -> Tensor:
        return self.predictor(self.project(x))

    def configure_optimizers(self):
        optimizer = getattr(optim, self.hparams.get("optimizer", "Adam"))
        weight_decay = self.hparams.get("weight_decay", 1e-6)
        param_dicts = [
            {
                "params": chain(
                    self.extractor.parameters(),
                    self.projector.parameters(),
                    self.predictor.parameters(),
                ),
                "lr": self.hparams.get("lr", 3e-4),
            },
            # {
            #     "params": self.linear.parameters(),
            #     "lr": self.hparams.get("linear_lr", 1e-3),
            # },
        ]
        return optimizer(param_dicts, weight_decay=weight_decay)

    def training_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x, y, *_ = batch
        with torch.no_grad():
            x1, x2 = self.augment(x), self.augment(x)
            targ1 = self.target_projector(self.target_extractor(x1))
            targ2 = self.target_projector(self.target_extractor(x2))
        pred1, pred2 = self.predict(x1), self.predict(x2)
        loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))

        if self.hparams.get("train_classifier", False):
            with torch.no_grad():
                extracted = self.forward(x).detach()
            pred = torch.log_softmax(self.linear(extracted), dim=-1)
            # Grab samples that have a label.
            mask = (y >= 0)
            loss += f.nll_loss(pred[mask], y[mask])

        return {"loss": loss, "log": {"train_loss": loss}}

    def on_zero_grad(self, *_):
        self.update_targets()

    @torch.no_grad()
    def validation_step(self, batch, *_) -> Dict[str, Union[Tensor, Dict]]:
        x, y, *_ = batch
        x1, x2 = self.augment(x), self.augment(x)
        targ1 = self.target_projector(self.target_extractor(x1))
        targ2 = self.target_projector(self.target_extractor(x2))
        pred1, pred2 = self.predict(x1), self.predict(x2)
        loss = torch.mean(normalized_mse(pred1, targ2) + normalized_mse(pred2, targ1))

        if self.hparams.get("train_classifier", False):
            extracted = self.forward(x)
            pred = torch.log_softmax(self.linear(extracted), dim=-1)
            loss += f.nll_loss(pred, y)

            metric_fns = self.hparams.get("metrics", ())
            metrics = {m.__name__: m(pred, y) for m in metric_fns}
        else:
            metrics = {}

        return {"loss": loss, "log": {"val_loss": loss, **metrics}}
