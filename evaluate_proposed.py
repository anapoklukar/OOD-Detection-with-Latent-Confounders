import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets import get_dataloaders
from src.vae import VAEClassifier
from src.ood_scores import energy_score
from src.metrics import compute_ood_metrics


def extract_train_latents(model, loader, device):
    model.eval()
    all_mu = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extract train latents"):
            images = images.to(device)
            mu, logvar = model.encode(images)
            all_mu.append(mu.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_mu), np.concatenate(all_labels)


def fit_classwise_mahalanobis(train_z, train_labels, num_classes=10, eps=1e-5):
    class_means = []

    for c in range(num_classes):
        z_c = train_z[train_labels == c]
        class_means.append(z_c.mean(axis=0))

    class_means = np.stack(class_means)

    centered = []
    for c in range(num_classes):
        z_c = train_z[train_labels == c]
        centered.append(z_c - class_means[c])

    centered = np.concatenate(centered)

    cov = np.cov(centered, rowvar=False)
    cov += eps * np.eye(cov.shape[0])
    precision = np.linalg.inv(cov)

    return class_means, precision


def classwise_mahalanobis(z, class_means, precision):
    scores = []

    for sample in z:
        distances = []
        for mean in class_means:
            diff = sample - mean
            dist = diff @ precision @ diff.T
            distances.append(dist)
        scores.append(min(distances))

    return np.array(scores)


def collect_all_scores(model, loader, device, class_means, precision, num_samples=20):
    model.eval()

    energy_scores = []
    latent_scores = []
    instability_scores = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Collect scores"):
            images = images.to(device)

            mu, logvar = model.encode(images)
            z = model.reparameterize(mu, logvar)
            logits = model.classify(z)

            # 1. Energy score
            energy = energy_score(logits).cpu().numpy()

            # 2. Negative classwise latent Mahalanobis
            latent_distance = classwise_mahalanobis(
                mu.cpu().numpy(),
                class_means,
                precision
            )
            latent = -latent_distance

            # 3. Instability score
            probs_list = []

            for _ in range(num_samples):
                z_sample = model.reparameterize(mu, logvar)
                logits_sample = model.classify(z_sample)
                probs = F.softmax(logits_sample, dim=1)
                probs_list.append(probs.unsqueeze(0))

            probs_stack = torch.cat(probs_list, dim=0)
            instability = torch.var(probs_stack, dim=0).mean(dim=1).cpu().numpy()

            energy_scores.extend(energy)
            latent_scores.extend(latent)
            instability_scores.extend(instability)

    return {
        "Energy": np.array(energy_scores),
        "Latent": np.array(latent_scores),
        "Instability": np.array(instability_scores),
    }


def zscore_using_id(id_scores, test_scores):
    mean = id_scores.mean()
    std = id_scores.std() + 1e-8
    return (test_scores - mean) / std


def evaluate_pair(id_scores, ood_scores):
    y_true = np.concatenate([
        np.zeros(len(id_scores)),
        np.ones(len(ood_scores)),
    ])
    scores = np.concatenate([id_scores, ood_scores])
    return compute_ood_metrics(y_true, scores)


def evaluate_combinations(id_dict, ood_dict, dataset_name):
    print(f"\n===== CIFAR-10 vs {dataset_name} =====")

    # Normalize each score using ID statistics only
    norm_id = {}
    norm_ood = {}

    for key in id_dict:
        norm_id[key] = zscore_using_id(id_dict[key], id_dict[key])
        norm_ood[key] = zscore_using_id(id_dict[key], ood_dict[key])

    combinations = {
        "Energy": ["Energy"],
        "Latent": ["Latent"],
        "Instability": ["Instability"],
        "Energy + Latent": ["Energy", "Latent"],
        "Energy + Instability": ["Energy", "Instability"],
        "Latent + Instability": ["Latent", "Instability"],
        "Energy + Latent + Instability": ["Energy", "Latent", "Instability"],
    }

    for name, keys in combinations.items():
        id_combined = sum(norm_id[k] for k in keys) / len(keys)
        ood_combined = sum(norm_ood[k] for k in keys) / len(keys)

        metrics = evaluate_pair(id_combined, ood_combined)

        print(
            f"{name:32s} "
            f"AUROC={metrics['AUROC']:.4f} "
            f"AUPR={metrics['AUPR']:.4f} "
            f"FPR95={metrics['FPR95']:.4f}"
        )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    loaders = get_dataloaders(batch_size=128)

    model = VAEClassifier(latent_dim=128, num_classes=10).to(device)
    model.load_state_dict(
        torch.load("models/vae_classifier.pth", map_location=device)
    )

    train_z, train_labels = extract_train_latents(model, loaders["train"], device)
    class_means, precision = fit_classwise_mahalanobis(train_z, train_labels)

    id_scores = collect_all_scores(
        model, loaders["cifar10_test"], device, class_means, precision
    )

    svhn_scores = collect_all_scores(
        model, loaders["svhn"], device, class_means, precision
    )

    cifar100_scores = collect_all_scores(
        model, loaders["cifar100"], device, class_means, precision
    )

    evaluate_combinations(id_scores, svhn_scores, "SVHN")
    evaluate_combinations(id_scores, cifar100_scores, "CIFAR-100")


if __name__ == "__main__":
    main()