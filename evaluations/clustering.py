import numpy as np
from sklearn import metrics


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def inverse_purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)


def harmonic_purity_score(y_true, y_pred):
    purity = purity_score(y_true, y_pred)
    inverse_purity = inverse_purity_score(y_true, y_pred)
    return (2.0 * purity * inverse_purity) / (purity + inverse_purity + 1e-12)


def clustering_metric(labels, preds):
    metrics_func = [
        {
            'name': 'Purity',
            'method': purity_score
        },
        {
            'name': 'InversePurity',
            'method': inverse_purity_score
        },
        {
            'name': 'HarmonicPurity',
            'method': harmonic_purity_score
        },
        {
            'name': 'NMI',
            'method': metrics.cluster.normalized_mutual_info_score
        },
        {
            'name': 'ARI',
            'method': metrics.adjusted_rand_score
        },
    ]

    results = dict()
    for func in metrics_func:
        results[func['name']] = func['method'](labels, preds)

    return results


def evaluate_clustering(theta, labels):
    preds = np.argmax(theta, axis=1)
    return clustering_metric(labels, preds)
