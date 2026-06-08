import torch
import torch.nn.functional as F


def msp_score(logits):
    probs = F.softmax(logits, dim=1)
    max_probs = probs.max(dim=1).values
    return -max_probs


def energy_score(logits):
    return -torch.logsumexp(logits, dim=1)