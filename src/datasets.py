import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataloaders(data_root="./data", batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    cifar10_train_full = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )

    cifar10_test = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )

    cifar100_test = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )

    svhn_test = datasets.SVHN(
        root=data_root,
        split="test",
        download=True,
        transform=transform
    )

    cifar10_train, cifar10_val = random_split(
        cifar10_train_full,
        [45000, 5000],
        generator=torch.Generator().manual_seed(42)
    )

    return {
        "train": DataLoader(cifar10_train, batch_size=batch_size, shuffle=True),
        "val": DataLoader(cifar10_val, batch_size=batch_size, shuffle=False),
        "cifar10_test": DataLoader(cifar10_test, batch_size=batch_size, shuffle=False),
        "cifar100": DataLoader(cifar100_test, batch_size=batch_size, shuffle=False),
        "svhn": DataLoader(svhn_test, batch_size=batch_size, shuffle=False),
    }