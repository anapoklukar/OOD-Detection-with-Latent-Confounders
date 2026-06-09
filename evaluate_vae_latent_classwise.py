import numpy as np
import torch
from tqdm import tqdm

from src.datasets import get_dataloaders
from src.vae import VAEClassifier
from src.metrics import compute_ood_metrics


def extract_latents(model, loader, device):
    model.eval()
    all_mu = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            mu, logvar = model.encode(images)

            all_mu.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_mu, axis=0), np.concatenate(all_labels, axis=0)


def fit_classwise_mahalanobis(train_z, train_labels, num_classes=10, eps=1e-5):
    class_means = []

    for c in range(num_classes):
        z_c = train_z[train_labels == c]
        class_means.append(z_c.mean(axis=0))

    class_means = np.stack(class_means, axis=0)

    centered = []

    for c in range(num_classes):
        z_c = train_z[train_labels == c]
        centered.append(z_c - class_means[c])

    centered = np.concatenate(centered, axis=0)

    covariance = np.cov(centered, rowvar=False)
    covariance += eps * np.eye(covariance.shape[0])

    precision = np.linalg.inv(covariance)

    return class_means, precision


def classwise_mahalanobis_score(z, class_means, precision):
    scores = []

    for sample in z:
        distances = []

        for mean in class_means:
            diff = sample - mean
            dist = diff @ precision @ diff.T
            distances.append(dist)

        scores.append(min(distances))

    return np.array(scores)


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
    model.load_state_dict(
        torch.load("models/vae_classifier.pth", map_location=device)
    )

    print("Extracting training latent vectors...")
    train_z, train_labels = extract_latents(model, loaders["train"], device)

    print("Fitting classwise latent Mahalanobis...")
    class_means, precision = fit_classwise_mahalanobis(
        train_z,
        train_labels,
        num_classes=10
    )

    print("Extracting test/OOD latent vectors...")
    id_z, _ = extract_latents(model, loaders["cifar10_test"], device)
    svhn_z, _ = extract_latents(model, loaders["svhn"], device)
    cifar100_z, _ = extract_latents(model, loaders["cifar100"], device)

    print("Computing classwise Mahalanobis scores...")
    id_scores = classwise_mahalanobis_score(id_z, class_means, precision)
    svhn_scores = classwise_mahalanobis_score(svhn_z, class_means, precision)
    cifar100_scores = classwise_mahalanobis_score(cifar100_z, class_means, precision)

    print("Mean scores:")
    print("ID mean:", np.mean(id_scores))
    print("SVHN mean:", np.mean(svhn_scores))
    print("CIFAR100 mean:", np.mean(cifar100_scores))

    print("\nClasswise Latent Mahalanobis: CIFAR-10 vs SVHN")
    print(evaluate_pair(id_scores, svhn_scores))

    print("\nClasswise Latent Mahalanobis: CIFAR-10 vs CIFAR-100")
    print(evaluate_pair(id_scores, cifar100_scores))

    print("\nNegative Classwise Latent Mahalanobis: CIFAR-10 vs SVHN")
    print(evaluate_pair(-id_scores, -svhn_scores))

    print("\nNegative Classwise Latent Mahalanobis: CIFAR-10 vs CIFAR-100")
    print(evaluate_pair(-id_scores, -cifar100_scores))


if __name__ == "__main__":
    main()