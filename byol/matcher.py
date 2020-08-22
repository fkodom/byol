from tqdm import tqdm
import torch
from torch import nn, Tensor
import torch.nn.functional as f
from torch.utils.data import DataLoader, Dataset


def normalized_mse(x: Tensor, y: Tensor) -> Tensor:
    x = f.normalize(x.flatten(start_dim=1), dim=1)
    y = f.normalize(y.flatten(start_dim=1), dim=1)
    return (x - y).pow(2).sum(dim=1)


class ImageMatcher:
    def __init__(self, extractor: nn.Module, dataset: Dataset):
        super().__init__()
        if torch.cuda.is_available():
            extractor = nn.DataParallel(extractor)
        self.extractor = extractor
        self.dataset = dataset

    def _features(self, images: Tensor) -> Tensor:
        return torch.cat(
            [v.flatten(start_dim=1) for v in self.extractor(images).values()], dim=1
        )

    @torch.no_grad()
    def get_matches(
        self, image: Tensor, n: int = 5, batch_size: int = 32, verbose: bool = False,
    ) -> Tensor:
        v = self._features(image.unsqueeze(0))
        loader = tqdm(DataLoader(self.dataset, batch_size=32), disable=not verbose)
        distances = torch.cat(
            [normalized_mse(v, self._features(x[0])) for x in loader], dim=0,
        )
        idx = torch.topk(distances, k=n + 1, largest=False).indices[1:]
        return torch.stack([self.dataset[i][0] for i in idx], dim=0)
