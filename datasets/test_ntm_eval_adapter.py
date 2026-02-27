from pathlib import Path
import os

import numpy as np

os.chdir(Path(__file__).resolve().parents[1])
from evaluation import ntm_eval_adapter as eva


def main():
    top_words = [
        ["apple", "banana", "carrot"],
        ["apple", "desk", "chair"],
    ]
    td = eva.compute_topic_diversity(top_words)
    assert 0.0 <= td <= 1.0

    theta = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]], dtype=np.float32)
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    cluster = eva.evaluate_clustering(theta, labels)
    for key in ["Purity", "InversePurity", "HarmonicPurity", "NMI", "ARI"]:
        assert key in cluster

    cls = eva.evaluate_classification(theta, theta, labels, labels, tune=False)
    assert "acc" in cls and "macro-F1" in cls
    assert cls["acc"] >= 0.5
    print("PASS")


if __name__ == "__main__":
    main()
