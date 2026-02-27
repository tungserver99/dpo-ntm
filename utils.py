from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import torch
import numpy as np
from pathlib import Path


def sparse2dense(input_matrix):
    if sparse.isspmatrix(input_matrix):
        input_matrix = input_matrix.toarray()
    input_matrix = input_matrix.astype('float32')
    return input_matrix


def get_word_embeddings(words, embedding_model):
    word_embeddings = []
    remove_idxs = []
    for i in range(len(words)):
        try:
            word_embedding = embedding_model[words[i].strip().lower()]
            word_embeddings.append(word_embedding)
        except:
            remove_idxs.append(i)

    return word_embeddings, remove_idxs


def construction_cost(topic_word_tm, topic_word_llm, embedding_model):
    word_embeddings_llm, remove_idxs_llm = get_word_embeddings(topic_word_llm, embedding_model)
    if len(word_embeddings_llm) == 0:
        print('All llm words are removed!')
        print(topic_word_llm)
        quit()

    word_embeddings_tm, remove_idxs_tm = get_word_embeddings(topic_word_tm, embedding_model)
    if len(word_embeddings_tm) == 0:
        print('All topic words are removed!')
        print(topic_word_tm)
        quit()

    # construct cost matrix
    cost_M = 1 - cosine_similarity(word_embeddings_tm, word_embeddings_llm)
    cost_M = torch.from_numpy(cost_M).to(torch.float64).cuda()

    return cost_M, remove_idxs_llm


class DatasetEmbeddingModel:
    def __init__(self, vocab, embeddings):
        self._emb = {}
        for i, word in enumerate(vocab):
            key = word.strip().lower()
            if key and key not in self._emb:
                self._emb[key] = embeddings[i]

    def __getitem__(self, word):
        key = word.strip().lower()
        if key not in self._emb:
            raise KeyError(word)
        return self._emb[key]


def _dataset_dir(dataset, datasets_root="datasets"):
    primary = Path(datasets_root) / dataset
    if primary.exists():
        return primary

    fallback = Path("LLM-ITL") / datasets_root / dataset
    if fallback.exists():
        return fallback

    return primary


def load_dataset_word_embeddings(dataset, datasets_root="datasets"):
    emb_path = _dataset_dir(dataset, datasets_root) / "word_embeddings.npy"
    return np.load(emb_path).astype("float32")


def load_dataset_vocab(dataset, datasets_root="datasets"):
    voc_path = _dataset_dir(dataset, datasets_root) / "voc.txt"
    return voc_path.read_text(encoding="utf-8", errors="ignore").split()


def load_dataset_embedding_model(dataset, datasets_root="datasets"):
    vocab = load_dataset_vocab(dataset, datasets_root)
    embeddings = load_dataset_word_embeddings(dataset, datasets_root)
    if embeddings.shape[0] != len(vocab):
        raise ValueError(
            f"Vocab/embedding size mismatch for {dataset}: "
            f"{len(vocab)} vs {embeddings.shape[0]}"
        )
    return DatasetEmbeddingModel(vocab, embeddings)
