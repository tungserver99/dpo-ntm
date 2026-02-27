from pathlib import Path
import os

os.chdir(Path(__file__).resolve().parents[1])
from utils import load_dataset_embedding_model, load_dataset_word_embeddings, load_dataset_vocab


def main():
    dataset = "BBC_new"
    emb = load_dataset_word_embeddings(dataset)
    assert emb.ndim == 2
    assert emb.shape[0] > 0
    assert emb.shape[1] > 0

    model = load_dataset_embedding_model(dataset)
    vocab = load_dataset_vocab(dataset)
    _ = model[vocab[0]]

    print("PASS")


if __name__ == "__main__":
    main()
