import torch
from torch import nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.datasets import STL10
from torchvision.transforms import ToTensor

from byol import BYOL


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_MSG = """[Epoch {epoch}]
    Loss: {loss:.3f}
    Acc: {acc:.3f}"""

TRAIN_DATASET = STL10(root="data", split="train", download=True, transform=ToTensor())
TEST_DATASET = STL10(root="data", split="test", download=True, transform=ToTensor())


def data_loader(train: bool = True, batch_size: int = 128) -> DataLoader:
    dataset = TRAIN_DATASET if train else TEST_DATASET
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=True)


def evaluate(resnet: nn.Module, epochs: int = 3) -> float:
    resnet = resnet.to(DEVICE)
    optimizer = torch.optim.Adam(getattr(resnet, "fc").parameters(), lr=1e-4)

    for epoch in range(epochs):
        for (x, y) in data_loader(train=True):
            optimizer.zero_grad()
            loss = f.cross_entropy(resnet(x.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            optimizer.step()

    accuracy = 0.0
    test_loader = data_loader(train=False)
    for (x, y) in test_loader:
        with torch.no_grad():
            pred = resnet(x.to(DEVICE))
            accuracy += (pred.argmax(dim=-1) == y.to(DEVICE)).float().mean()

    return accuracy / len(test_loader)


def train(resnet: nn.Module, epochs: int = 10, batch_size: int = 128):
    learner = BYOL(resnet, image_size=(96, 96), hidden_layer="avgpool").to(DEVICE)
    # learner = BYOL(resnet, image_size=96, hidden_layer="avgpool").to(DEVICE)
    optimizer = torch.optim.Adam(learner.parameters(), lr=1e-4)

    for epoch in range(epochs):
        epoch_loss = 0.0
        train_loader = data_loader(train=True)

        for (x, _) in train_loader:
            optimizer.zero_grad()
            loss = learner(x.to(DEVICE))
            loss.backward()
            optimizer.step()
            learner.update_target()
            # learner.update_moving_average()
            epoch_loss += loss.item() / len(train_loader)

        print(EVAL_MSG.format(epoch=epoch, loss=epoch_loss, acc=evaluate(resnet)))


resnet = resnet50(pretrained=True)
print(resnet)
print("Pretrained Resnet50")
# print(EVAL_MSG.format(epoch="N/A", loss=0.0, acc=evaluate(resnet)))

train(resnet, epochs=25)
