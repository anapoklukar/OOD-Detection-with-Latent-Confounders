import torch
import torch.nn.functional as F
import numpy as np


def msp_score(logits):
    probs = F.softmax(logits, dim=1)
    max_probs = probs.max(dim=1).values
    return -max_probs


def energy_score(logits):
    return -torch.logsumexp(logits, dim=1)


def mahalanobis_score(features, class_means, precision):
    scores = []

    for feat in features:
        distances = []

        for mean in class_means:
            diff = feat - mean
            dist = diff @ precision @ diff.T
            distances.append(dist)

        scores.append(min(distances))

    return np.array(scores)