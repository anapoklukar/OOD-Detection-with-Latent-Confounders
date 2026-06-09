import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEClassifier(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # 4x4
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),    # 32x32
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def classify(self, z):
        return self.classifier(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        logits = self.classify(z)
        return recon, logits, mu, logvar, z


def vae_loss(recon, x, logits, labels, mu, logvar, beta=0.001, cls_weight=1.0):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    cls_loss = F.cross_entropy(logits, labels)

    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total_loss = recon_loss + beta * kl_loss + cls_weight * cls_loss

    return total_loss, recon_loss, kl_loss, cls_loss