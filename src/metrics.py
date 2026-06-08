import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def compute_fpr95(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    idx = np.argmin(np.abs(tpr - 0.95))
    return fpr[idx]


def compute_ood_metrics(y_true, scores):
    return {
        "AUROC": roc_auc_score(y_true, scores),
        "AUPR": average_precision_score(y_true, scores),
        "FPR95": compute_fpr95(y_true, scores),
    }