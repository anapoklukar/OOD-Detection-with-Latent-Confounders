import numpy as np
import torch
from tqdm import tqdm

from src.datasets import get_dataloaders
from src.vae import VAEClassifier
from src.metrics import compute_ood_metrics


def extract_latents(model, loader, device):
    model.eval()
    mus = []
    labels_all = []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            mu, logvar = model.encode(images)

            mus.append(mu.cpu().numpy())
            labels_all.append(labels.numpy())

    return np.concatenate(mus, axis=0), np.concatenate(labels_all, axis=0)


def fit_latent_mahalanobis(train_z, eps=1e-5):
    mean = train_z.mean(axis=0)
    centered = train_z - mean

    cov = np.cov(centered, rowvar=False)
    cov += eps * np.eye(cov.shape[0])

    precision = np.linalg.inv(cov)

    return mean, precision


def latent_mahalanobis_score(z, mean, precision):
    diff = z - mean
    scores = np.sum((diff @ precision) * diff, axis=1)
    return scores


def evaluate_pair(id_scores, ood_scores):
    y_true = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])
    return compute_ood_metrics(y_true, scores)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    loaders = get_dataloaders(batch_size=128)

    model = VAEClassifier(latent_dim=128, num_classes=10).to(device)
    model.load_state_dict(torch.load("models/vae_classifier.pth", map_location=device))

    print("Extracting train latent vectors...")
    train_z, _ = extract_latents(model, loaders["train"], device)

    print("Fitting latent Mahalanobis...")
    mean, precision = fit_latent_mahalanobis(train_z)

    print("Extracting test/OOD latent vectors...")
    id_z, _ = extract_latents(model, loaders["cifar10_test"], device)
    svhn_z, _ = extract_latents(model, loaders["svhn"], device)
    cifar100_z, _ = extract_latents(model, loaders["cifar100"], device)

    id_scores = latent_mahalanobis_score(id_z, mean, precision)
    svhn_scores = latent_mahalanobis_score(svhn_z, mean, precision)
    cifar100_scores = latent_mahalanobis_score(cifar100_z, mean, precision)

    print("\nLatent Mahalanobis: CIFAR-10 vs SVHN")
    print(evaluate_pair(id_scores, svhn_scores))

    print("\nLatent Mahalanobis: CIFAR-10 vs CIFAR-100")
    print(evaluate_pair(id_scores, cifar100_scores))


if __name__ == "__main__":
    main()