import numpy as np


def mahalanobis_score(features, class_means, precision):
    """
    Higher score = more likely OOD.
    """
    scores = []

    for feat in features:
        distances = []
        for mean in class_means:
            diff = feat - mean
            dist = diff @ precision @ diff.T
            distances.append(dist)

        scores.append(min(distances))

    return np.array(scores)