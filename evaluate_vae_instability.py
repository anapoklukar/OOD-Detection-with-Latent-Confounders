import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.datasets import get_dataloaders
from src.vae import VAEClassifier
from src.metrics import compute_ood_metrics


def instability_scores(model, loader, device, num_samples=20):
    model.eval()
    scores = []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            mu, logvar = model.encode(images)

            probs_list = []

            for _ in range(num_samples):
                z = model.reparameterize(mu, logvar)
                logits = model.classify(z)
                probs = F.softmax(logits, dim=1)
                probs_list.append(probs.unsqueeze(0))

            probs_stack = torch.cat(probs_list, dim=0)
            variance = torch.var(probs_stack, dim=0).mean(dim=1)

            scores.extend(variance.cpu().numpy())

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
    model.load_state_dict(torch.load("models/vae_classifier.pth", map_location=device))

    id_scores = instability_scores(model, loaders["cifar10_test"], device)
    svhn_scores = instability_scores(model, loaders["svhn"], device)
    cifar100_scores = instability_scores(model, loaders["cifar100"], device)

    print("\nInstability: CIFAR-10 vs SVHN")
    print(evaluate_pair(id_scores, svhn_scores))

    print("\nInstability: CIFAR-10 vs CIFAR-100")
    print(evaluate_pair(id_scores, cifar100_scores))


if __name__ == "__main__":
    main()