# OOD Detection with Latent Confounders

Course project for **Special Topics in Data Science, Spring 2026**.

Team members:

- Muhammad Shahzaib Khan
- Ana Poklukar

## Project Overview

This repository evaluates out-of-distribution (OOD) detection on image data. CIFAR-10 is treated as the in-distribution dataset, while SVHN and CIFAR-100 are used as OOD datasets.

The project compares standard OOD baselines against a proposed VAE-based multi-signal method.

## Methods

### Baselines

The baseline methods use a ResNet-18 classifier trained on CIFAR-10:

- **MSP**: maximum softmax probability
- **Energy**: energy score computed from classifier logits
- **Mahalanobis**: classwise feature-space Mahalanobis distance

### Proposed Method

The proposed method trains a VAE classifier on CIFAR-10. The model contains:

- an encoder that maps an image to latent parameters `mu` and `logvar`
- a stochastic latent variable `z` sampled using the reparameterization trick
- a decoder that reconstructs the image
- a classifier that predicts the CIFAR-10 class from `z`

The proposed OOD score combines three signals:

- **Energy**: confidence score from VAE classifier logits
- **Latent anomaly**: classwise Mahalanobis distance in VAE latent space
- **Instability**: prediction variance from repeated latent sampling

In `scripts/evaluate_proposed.py`, each score is z-score normalized using in-distribution statistics, then combined by equal averaging.

## Repository Structure

```text
src/datasets.py              Dataset loading for CIFAR-10, CIFAR-100, and SVHN
src/models.py                ResNet-18 baseline model
src/vae.py                   VAE classifier model and loss
src/ood_scores.py            MSP, Energy, and Mahalanobis score functions
src/metrics.py               AUROC, AUPR, and FPR95 metrics

scripts/check_data.py        Verify dataset loading
scripts/train_classifier.py  Train ResNet-18 baseline
scripts/evaluate_baselines.py
                             Evaluate ResNet MSP and Energy baselines
scripts/evaluate_mahalanobis.py
                             Evaluate ResNet feature Mahalanobis baseline

scripts/train_vae_classifier.py
                             Train VAE classifier
scripts/evaluate_vae_energy.py
                             Evaluate VAE MSP and Energy scores
scripts/evaluate_vae_latent.py
                             Evaluate VAE global latent Mahalanobis score
scripts/evaluate_vae_latent_classwise.py
                             Evaluate VAE classwise latent Mahalanobis score
scripts/evaluate_vae_instability.py
                             Evaluate VAE prediction instability score
scripts/evaluate_proposed.py Evaluate individual and fused proposed scores
```

Generated files are created during training and evaluation:

```text
data/       downloaded datasets
models/     trained model checkpoints
```

## Data

The datasets are public and are downloaded automatically through `torchvision`:

- CIFAR-10: in-distribution training and test data
- SVHN: far-OOD test data
- CIFAR-100: near-OOD test data

No private credentials or restricted datasets are required.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If installing PyTorch with CUDA, use the command recommended by the official PyTorch installer for your GPU/CUDA version.

## Reproducing the Main Pipeline

To run the full training and evaluation pipeline with one command:

```bash
bash scripts/run_main_pipeline.sh
```

This downloads the public datasets, trains both models, and prints the main metric tables. Full training can take a while, so the steps can also be run individually as shown below.

First, check that the datasets can be loaded:

```bash
python3 scripts/check_data.py
```

Train the ResNet baseline:

```bash
python3 scripts/train_classifier.py
```

This saves:

```text
models/cifar10_resnet18.pth
```

Evaluate baseline methods:

```bash
python3 scripts/evaluate_baselines.py
python3 scripts/evaluate_mahalanobis.py
```

Train the VAE classifier:

```bash
python3 scripts/train_vae_classifier.py
```

This saves:

```text
models/vae_classifier.pth
```

Evaluate individual VAE signals:

```bash
python3 scripts/evaluate_vae_energy.py
python3 scripts/evaluate_vae_latent.py
python3 scripts/evaluate_vae_latent_classwise.py
python3 scripts/evaluate_vae_instability.py
```

Evaluate the proposed fused method:

```bash
python3 scripts/evaluate_proposed.py
```

## Evaluation

All evaluation scripts compare:

- CIFAR-10 test set vs SVHN
- CIFAR-10 test set vs CIFAR-100

The reported metrics are:

- **AUROC**
- **AUPR**
- **FPR95**

Higher AUROC and AUPR are better. Lower FPR95 is better.
