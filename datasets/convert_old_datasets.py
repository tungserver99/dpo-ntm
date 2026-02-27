import argparse
import pickle
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


def _read_lines(path: Path):
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _build_bow_dict(bow_csr: sp.csr_matrix, labels_2d: np.ndarray):
    tokens = []
    counts = []

    indptr = bow_csr.indptr
    indices = bow_csr.indices
    data = bow_csr.data
    for i in range(bow_csr.shape[0]):
        start = indptr[i]
        end = indptr[i + 1]
        tokens.append(indices[start:end].astype(np.int32, copy=True))
        counts.append(data[start:end].astype(np.int64, copy=True))

    return {"tokens": tokens, "counts": counts, "labels": labels_2d}


def convert_one(old_root: Path, new_root: Path, dataset_name: str):
    src = old_root / dataset_name
    if not src.exists():
        raise FileNotFoundError(f"Missing dataset folder: {src}")

    bow_train = sp.load_npz(src / "train_bow.npz").astype(np.float32).tocsr()
    bow_test = sp.load_npz(src / "test_bow.npz").astype(np.float32).tocsr()
    word_embeddings = sp.load_npz(src / "word_embeddings.npz").toarray().astype(np.float32)

    vocab = _read_lines(src / "vocab.txt")
    train_texts = _read_lines(src / "train_texts.txt")
    test_texts = _read_lines(src / "test_texts.txt")
    train_labels = np.loadtxt(src / "train_labels.txt", dtype=np.int64).reshape(-1, 1)
    test_labels = np.loadtxt(src / "test_labels.txt", dtype=np.int64).reshape(-1, 1)

    assert bow_train.shape[0] == len(train_texts) == train_labels.shape[0]
    assert bow_test.shape[0] == len(test_texts) == test_labels.shape[0]
    assert bow_train.shape[1] == bow_test.shape[1] == len(vocab)
    assert word_embeddings.shape[0] == len(vocab)

    ds_dir = new_root / dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)

    train_pkl = _build_bow_dict(bow_train, train_labels)
    test_pkl = {"test": _build_bow_dict(bow_test, test_labels), "labels": test_labels}

    with (ds_dir / "train.pkl").open("wb") as f:
        pickle.dump(train_pkl, f)
    with (ds_dir / "test.pkl").open("wb") as f:
        pickle.dump(test_pkl, f)
    (ds_dir / "voc.txt").write_text(" ".join(vocab), encoding="utf-8")
    np.save(ds_dir / "word_embeddings.npy", word_embeddings)

    voc_np = np.array(vocab, dtype=np.object_).reshape(1, -1)
    doc_train_np = np.array(train_texts, dtype=np.object_).reshape(-1, 1)
    doc_test_np = np.array(test_texts, dtype=np.object_).reshape(-1, 1)
    sio.savemat(
        new_root / f"{dataset_name}.mat",
        {
            "bow_train": bow_train,
            "bow_test": bow_test,
            "voc": voc_np,
            "label_train": train_labels,
            "label_test": test_labels,
            "doc_train": doc_train_np,
            "doc_test": doc_test_np,
        },
    )

    print(
        f"[OK] {dataset_name}: train={bow_train.shape}, "
        f"test={bow_test.shape}, vocab={len(vocab)}, emb={word_embeddings.shape}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["BBC_new", "NYT", "WOS_vocab_10k"],
        help="Dataset folder names under old_datasets/",
    )
    parser.add_argument("--old-root", default="LLM-ITL/old_datasets")
    parser.add_argument("--new-root", default="LLM-ITL/datasets")
    args = parser.parse_args()

    old_root = Path(args.old_root)
    new_root = Path(args.new_root)
    for dataset_name in args.datasets:
        convert_one(old_root, new_root, dataset_name)


if __name__ == "__main__":
    main()
