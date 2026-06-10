import _path  # noqa: F401
from src.datasets import get_dataloaders

loaders = get_dataloaders()

for name, loader in loaders.items():
    images, labels = next(iter(loader))
    print(name, images.shape, labels.shape)
