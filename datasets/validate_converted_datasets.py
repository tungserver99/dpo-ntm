import argparse
import pickle
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import sparse


def _dense_row(x):
    if sparse.isspmatrix(x):
        return np.asarray(x.toarray()).ravel()
    return np.asarray(x).ravel()


def validate_one(new_root: Path, dataset_name: str):
    ds_dir = new_root / dataset_name
    mat_path = new_root / f"{dataset_name}.mat"
    assert ds_dir.exists(), f"Missing folder: {ds_dir}"
    assert mat_path.exists(), f"Missing mat file: {mat_path}"

    required_files = ["train.pkl", "test.pkl", "voc.txt", "word_embeddings.npy"]
    for fn in required_files:
        assert (ds_dir / fn).exists(), f"Missing file: {ds_dir / fn}"

    data = sio.loadmat(mat_path)
    for key in ["bow_train", "bow_test", "voc", "label_train", "label_test"]:
        assert key in data, f"Missing key `{key}` in {mat_path}"

    bow_train = data["bow_train"]
    bow_test = data["bow_test"]
    labels_train = data["label_train"]
    labels_test = data["label_test"]
    vocab_mat = [v[0] for v in data["voc"].reshape(-1).tolist()]

    with (ds_dir / "train.pkl").open("rb") as f:
        train_pkl = pickle.load(f)
    with (ds_dir / "test.pkl").open("rb") as f:
        test_pkl = pickle.load(f)

    vocab_txt = (ds_dir / "voc.txt").read_text(encoding="utf-8", errors="ignore").split()
    embeddings = np.load(ds_dir / "word_embeddings.npy")

    assert bow_train.shape[1] == bow_test.shape[1] == len(vocab_txt)
    assert len(vocab_mat) == len(vocab_txt)
    assert embeddings.shape[0] == len(vocab_txt)
    assert labels_train.shape[0] == bow_train.shape[0]
    assert labels_test.shape[0] == bow_test.shape[0]
    assert len(train_pkl["tokens"]) == len(train_pkl["counts"]) == bow_train.shape[0]
    assert len(test_pkl["test"]["tokens"]) == len(test_pkl["test"]["counts"]) == bow_test.shape[0]
    assert train_pkl["labels"].shape == labels_train.shape
    assert test_pkl["labels"].shape == labels_test.shape

    for i in [0, bow_train.shape[0] // 2, bow_train.shape[0] - 1]:
        row = _dense_row(bow_train[i])
        idx = np.asarray(train_pkl["tokens"][i])
        cnt = np.asarray(train_pkl["counts"][i])
        assert np.array_equal(idx, np.where(row > 0)[0]), f"train tokens mismatch at row {i}"
        assert np.allclose(cnt, row[idx]), f"train counts mismatch at row {i}"

    for i in [0, bow_test.shape[0] // 2, bow_test.shape[0] - 1]:
        row = _dense_row(bow_test[i])
        idx = np.asarray(test_pkl["test"]["tokens"][i])
        cnt = np.asarray(test_pkl["test"]["counts"][i])
        assert np.array_equal(idx, np.where(row > 0)[0]), f"test tokens mismatch at row {i}"
        assert np.allclose(cnt, row[idx]), f"test counts mismatch at row {i}"

    print(f"[PASS] {dataset_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["BBC_new", "NYT", "WOS_vocab_10k"],
        help="Dataset names under LLM-ITL/datasets",
    )
    parser.add_argument("--new-root", default="LLM-ITL/datasets")
    args = parser.parse_args()

    new_root = Path(args.new_root)
    for dataset_name in args.datasets:
        validate_one(new_root, dataset_name)


if __name__ == "__main__":
    main()
