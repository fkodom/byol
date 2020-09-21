# BYOL: Bootstrap Your Own Latent

PyTorch implementation of [BYOL](https://arxiv.org/abs/2006.07733): a fantastically simple method for self-supervised image representation learning with SOTA performance.  Strongly influenced and inspired by [this Github repo](https://github.com/lucidrains/byol-pytorch), but with a few notable differences:
1. Enables **multi-GPU** training in PyTorch Lightning.
2. (Optionally) Automatically trains a linear classifier, and logs its accuracy after each epoch.
3. All functions and classes are fully type annotated for better usability/hackability with Python>=3.6.


## TO DO
* Enable mixed-precision training in PyTorch Lightning.  `kornia.augmentation.RandomResizedCrop` currently doesn't support this.  I'll need to ensure that our implementation is sufficiently performant, so it doesn't inadvertently slow down training.
