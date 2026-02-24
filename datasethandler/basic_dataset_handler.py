import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.sparse

from . import file_utils


class DatasetHandler(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return {
            "data": self.data[idx],
            "idx": idx,
        }


class RawDatasetHandler:
    def __init__(self, docs, preprocessing, batch_size=200, device="cpu", as_tensor=False, contextual_embed=False):
        rst = preprocessing.preprocess(docs)
        self.train_data = rst["train_bow"]
        self.train_texts = rst["train_texts"]
        self.vocab = rst["vocab"]
        self.vocab_size = len(self.vocab)

        if as_tensor:
            self.train_data = torch.from_numpy(self.train_data).float().to(device)
            self.train_dataloader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)


class BasicDatasetHandler:
    def __init__(
        self,
        dataset_dir,
        batch_size=200,
        read_labels=False,
        device="cpu",
        as_tensor=False,
        contextual_embed=False,
    ):
        # keep these args for backward compatibility in callers
        _ = contextual_embed
        self.dataset_dir = dataset_dir
        self.load_data(dataset_dir, read_labels)
        self.vocab_size = len(self.vocab)

        print("===>train_size: ", self.train_bow.shape[0])
        print("===>test_size: ", self.test_bow.shape[0])
        print("===>vocab_size: ", self.vocab_size)
        print("===>average length: {:.3f}".format(self.train_bow.sum(1).sum() / self.train_bow.shape[0]))

        if as_tensor:
            self.train_data = torch.from_numpy(self.train_bow).to(device)
            self.test_data = torch.from_numpy(self.test_bow).to(device)

            train_dataset = DatasetHandler(self.train_data)
            test_dataset = DatasetHandler(self.test_data)

            self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def load_data(self, path, read_labels):
        self.train_bow = scipy.sparse.load_npz(f"{path}/train_bow.npz").toarray().astype("float32")
        self.test_bow = scipy.sparse.load_npz(f"{path}/test_bow.npz").toarray().astype("float32")
        self.pretrained_WE = scipy.sparse.load_npz(f"{path}/word_embeddings.npz").toarray().astype("float32")

        self.train_texts = file_utils.read_text(f"{path}/train_texts.txt")
        self.test_texts = file_utils.read_text(f"{path}/test_texts.txt")

        if read_labels:
            self.train_labels = np.loadtxt(f"{path}/train_labels.txt", dtype=int)
            self.test_labels = np.loadtxt(f"{path}/test_labels.txt", dtype=int)

        self.vocab = file_utils.read_text(f"{path}/vocab.txt")
