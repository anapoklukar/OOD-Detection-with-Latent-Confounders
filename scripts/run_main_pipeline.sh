#!/usr/bin/env bash
set -euo pipefail

python3 scripts/check_data.py
python3 scripts/train_classifier.py
python3 scripts/evaluate_baselines.py
python3 scripts/evaluate_mahalanobis.py
python3 scripts/train_vae_classifier.py
python3 scripts/evaluate_vae_energy.py
python3 scripts/evaluate_vae_latent_classwise.py
python3 scripts/evaluate_vae_instability.py
python3 scripts/evaluate_proposed.py
