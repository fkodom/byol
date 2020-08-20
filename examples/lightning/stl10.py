from os import cpu_count

import torch
from torch import Tensor
from torch.cuda import device_count
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

from torchvision.models import resnet50
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor

# from byol import ByolLightningModule
from byol.byol_ import BYOL as ByolLightningModule

TRAIN_DATASET = STL10(root="data", split="train", download=True, transform=ToTensor())
TEST_DATASET = STL10(root="data", split="test", download=True, transform=ToTensor())
IMAGE_SIZE = (96, 96)


def accuracy(pred: Tensor, labels: Tensor) -> float:
    return (pred.argmax(dim=-1) == labels).float().mean()


def train(
    experiment: str = "stl10",
    run: str = None,
    gpus: int = device_count(),
    precision: int = 32,
    model: str = "resnet50",
    optimizer: str = "Adam",
    monitor: str = "accuracy",
    lr: float = 1e-4,
    epochs: int = 25,
    batch_size: int = 128,
    num_workers: int = cpu_count(),
    checkpoint: str = None,
    weights: str = None,
    debug: bool = False,
):
    """Executes a complete training loop, and optionally logs metrics to MLFlow.

    Parameters
    ----------
    experiment: (str) Name of the experiment (can have multiple runs per experiment)
    run: (str) Name of this run
    gpus: (int) Number of GPUs to use for training/validation
    model: (str) Name of the backbone model to use
    optimizer: (str) Name of the PyTorch optimizer to use for training
    monitor: (str) Performance metric used to determine top-k models to save
    lr: (float) Learning rate for all parameters *except the backbone*
    epochs: (int) Number of training epochs to perform.  One epoch trains on
        each example in the dataset exactly once.
    batch_size: (int) Size of each training/validation batch to use
    num_workers: (int) Number of worker threads used for DataLoaders.  (If you
        encounter errors with swap memory in Docker, set this to 0!)
    checkpoint: (str) The model checkpoint to load from file, if any
    weights: (str) Checkpoint file containing model weights to load.  Does not
        load the entire optimization state dictionary!
    debug: (bool) If True, performs a very short training sequence for debugging
        purposes, and no data is logged to MLFlow.
    """
    net = ByolLightningModule(
        model=getattr(models, model)(pretrained=True),
        train_dataset=TRAIN_DATASET,
        val_dataset=TEST_DATASET,
        image_size=IMAGE_SIZE,
        optimizer=optimizer,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    if checkpoint is not None:
        weights = args.checkpoint
    if weights is not None:
        net.load_state_dict(torch.load(weights)["state_dict"])

    if debug:
        logger, checkpoint_callback = None, None
    else:
        logger = MLFlowLogger(experiment, tags={MLFLOW_RUN_NAME: run})
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            prefix=f"{experiment}_{run}_", filepath="checkpoints", monitor=monitor,
        )

    pl.Trainer(
        gpus=gpus,
        precision=precision,
        amp_level="O1",
        distributed_backend="ddp",
        max_epochs=epochs,
        gradient_clip_val=1.0,
        resume_from_checkpoint=checkpoint,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        weights_summary=None,
        fast_dev_run=debug,
    ).fit(net)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", "-x", default="detr")
    parser.add_argument("--run", "-r", default=None)
    parser.add_argument("--coco_dir", default="coco")
    parser.add_argument("--gpus", default=device_count(), type=int)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--monitor", default="val_loss")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--epochs", "-e", default=10, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--workers", default=cpu_count(), type=int)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--surrogate", action="store_true")
    args = parser.parse_args()

    train(
        experiment=args.experiment,
        run=args.run,
        gpus=args.gpus,
        precision=args.precision,
        model=args.model,
        optimizer=args.optimizer,
        monitor=args.monitor,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.workers,
        checkpoint=args.checkpoint,
        weights=args.weights,
        debug=args.debug,
    )
