import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.sparse
import scipy.io
from sentence_transformers import SentenceTransformer
from . import file_utils
import os


def load_contextual_embed(texts, device, model_name="all-mpnet-base-v2", show_progress_bar=True):
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, show_progress_bar=show_progress_bar)
    return embeddings


class DatasetHandler(Dataset):
    def __init__(self, data, contextual_embed=None):
        self.data = data
        self.contextual_embed = None
        if contextual_embed is not None:
            assert data.shape[0] == contextual_embed.shape[0], "Data and contextual embeddings should have the same number of samples"
            self.contextual_embed = contextual_embed

    def __len__(self):
        # Update this according to your data size
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.contextual_embed is None:
            return {
                'data': self.data[idx]
            }

        return {
            'data': self.data[idx],
            'contextual_embed': self.contextual_embed[idx]
        }


class RawDatasetHandler:
    def __init__(self, docs, preprocessing, batch_size=200, device='cpu', as_tensor=False, contextual_embed=False):

        rst = preprocessing.preprocess(docs)
        self.train_data = rst['train_bow']
        self.train_texts = rst['train_texts']
        self.vocab = rst['vocab']

        self.vocab_size = len(self.vocab)

        if contextual_embed:
            self.train_contextual_embed = load_contextual_embed(
                self.train_texts, device)
            self.contextual_embed_size = self.train_contextual_embed.shape[1]

        if as_tensor:
            if contextual_embed:
                self.train_data = np.concatenate(
                    (self.train_data, self.train_contextual_embed), axis=1)

            self.train_data = torch.from_numpy(
                self.train_data).float().to(device)
            self.train_dataloader = DataLoader(
                self.train_data, batch_size=batch_size, shuffle=True)


class BasicDatasetHandler:
    def __init__(self, dataset_dir, batch_size=200, read_labels=False, device='cpu', as_tensor=False, contextual_embed=False, plm_model="all-mpnet-base-v2"):
        # train_bow: NxV
        # test_bow: Nxv
        # word_emeddings: VxD
        # vocab: V, ordered by word id.

        self.load_data(dataset_dir, read_labels)
        self.vocab_size = len(self.vocab)
        self.plm_model = plm_model

        print("===>train_size: ", self.train_bow.shape[0])
        print("===>test_size: ", self.test_bow.shape[0])
        print("===>vocab_size: ", self.vocab_size)
        print("===>average length: {:.3f}".format(
            self.train_bow.sum(1).sum() / self.train_bow.shape[0]))

        if contextual_embed:
            if os.path.isfile(os.path.join(dataset_dir, 'with_bert', 'train_bert.npz')):
                self.train_contextual_embed = np.load(os.path.join(
                    dataset_dir, 'with_bert', 'train_bert.npz'))['arr_0']
            else:
                self.train_contextual_embed = load_contextual_embed(
                    self.train_texts, device, model_name=self.plm_model)

            if os.path.isfile(os.path.join(dataset_dir, 'with_bert', 'test_bert.npz')):
                self.test_contextual_embed = np.load(os.path.join(
                    dataset_dir, 'with_bert', 'test_bert.npz'))['arr_0']
            else:
                self.test_contextual_embed = load_contextual_embed(
                    self.test_texts, device, model_name=self.plm_model)

            self.contextual_embed_size = self.train_contextual_embed.shape[1]

        if as_tensor:
            # if not contextual_embed:  # to be fixed with an additional argument
            #     self.train_data = self.train_bow
            #     self.test_data = self.test_bow
            # else:
            #     self.train_data = np.concatenate((self.train_bow, self.train_contextual_embed), axis=1)
            #     self.test_data = np.concatenate((self.test_bow, self.test_contextual_embed), axis=1)
            self.train_data = self.train_bow
            self.test_data = self.test_bow

            self.train_data = torch.from_numpy(self.train_data).to(device)
            self.test_data = torch.from_numpy(self.test_data).to(device)

            if contextual_embed:

                self.train_contextual_embed = torch.from_numpy(
                    self.train_contextual_embed).to(device)
                self.test_contextual_embed = torch.from_numpy(
                    self.test_contextual_embed).to(device)

                train_dataset = DatasetHandler(
                    self.train_data, self.train_contextual_embed)
                test_dataset = DatasetHandler(
                    self.test_data, self.test_contextual_embed)

                self.train_dataloader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True)
                self.test_dataloader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False)

            else:
                train_dataset = DatasetHandler(self.train_data)
                test_dataset = DatasetHandler(self.test_data)

                self.train_dataloader = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True)
                self.test_dataloader = DataLoader(
                    test_dataset, batch_size=batch_size, shuffle=False)

    def load_data(self, path, read_labels):

        self.train_bow = scipy.sparse.load_npz(
            f'{path}/train_bow.npz').toarray().astype('float32')
        self.test_bow = scipy.sparse.load_npz(
            f'{path}/test_bow.npz').toarray().astype('float32')
        self.pretrained_WE = scipy.sparse.load_npz(
            f'{path}/word_embeddings.npz').toarray().astype('float32')

        self.train_texts = file_utils.read_text(f'{path}/train_texts.txt')
        self.test_texts = file_utils.read_text(f'{path}/test_texts.txt')

        if read_labels:
            self.train_labels = np.loadtxt(
                f'{path}/train_labels.txt', dtype=int)
            self.test_labels = np.loadtxt(f'{path}/test_labels.txt', dtype=int)

        self.vocab = file_utils.read_text(f'{path}/vocab.txt')
        # self.train_bow = scipy.sparse.load_npz(
        #     os.path.join(path, 'train_bow.npz')).toarray().astype('float32')
        # self.test_bow = scipy.sparse.load_npz(
        #     os.path.join(path, 'test_bow.npz')).toarray().astype('float32')
        # self.pretrained_WE = scipy.sparse.load_npz(
        #     os.path.join(path, 'word_embeddings.npz')).toarray().astype('float32')

        # self.train_texts = file_utils.read_text(os.path.join(path, 'train_texts.txt'))
        # self.test_texts = file_utils.read_text(os.path.join(path, 'test_texts.txt'))

        # if read_labels:
        #     self.train_labels = np.loadtxt(
        #         os.path.join(path, 'train_labels.txt'), dtype=int)
        #     self.test_labels = np.loadtxt(os.path.join(path, 'test_labels.txt'), dtype=int)

        # self.vocab = file_utils.read_text(os.path.join(path, 'vocab.txt'))

class SubDataset(Dataset):
    def __init__(self, bows_np, sub_npz_path, device="cpu"):
        """
        bows_np: (N, V) numpy float32
        sub_npz_path: .../with_sub_bert/train_sub_bert.npz
                      arr_0 = E (total_subs, D), arr_1 = indptr (N+1,)
        """
        self.device = device
        self.bows = torch.from_numpy(bows_np).to(device)

        z = np.load(sub_npz_path)
        self.E = z["arr_0"]        # (total_subs, D) numpy
        self.indptr = z["arr_1"]   # (N+1,) numpy int64

        assert self.bows.shape[0] + 1 == self.indptr.shape[0], "indptr must be N+1"

    def __len__(self):
        return self.bows.shape[0]

    def __getitem__(self, i):
        # BoW tensor
        bow_i = self.bows[i]

        # Slice rows for doc i
        s, e = int(self.indptr[i]), int(self.indptr[i+1])
        # Make a list of 1D tensors (D,)
        sub_list = [torch.from_numpy(self.E[j]).to(self.device) for j in range(s, e)]

        return {
            "data": bow_i,                         # (V,)
            "sub_contextual_embed": sub_list       # list of (D,) tensors (len = num_subs_i)
        }

def collate_keep_lists(batch):
    # stack only 'data'; keep the sub lists as-is
    data = torch.stack([b["data"] for b in batch], dim=0)
    subs = [b["sub_contextual_embed"] for b in batch]  # list of lists
    return {"data": data, "sub_contextual_embed": subs}

def _npz_to_list_of_lists(npz_path, num_docs, device="cpu"):
    """
    Convert sub-embedding npz into list(list(tensor(D,))).
    Args:
        npz_path: path to .npz file with arr_0=E (total_subs, D), arr_1=indptr (N+1,)
        num_docs: number of documents (N)
    Returns:
        embeddings: list of length N
                    embeddings[i] = list of length num_sub_i
                                    each element is torch.FloatTensor (D,)
    """
    z = np.load(npz_path, mmap_mode="r")
    E = z["arr_0"]        # (total_subs, D)
    indptr = z["arr_1"]   # (N+1,)

    assert indptr.shape[0] == num_docs + 1, \
        f"indptr length {indptr.shape[0]} != N+1 = {num_docs+1}"

    perdoc = []
    for i in range(num_docs):
        s, e = int(indptr[i]), int(indptr[i+1])
        sub_list = [torch.from_numpy(E[j]).to(device) for j in range(s, e)]
        perdoc.append(sub_list)
    return perdoc


class SAEDatasetHandler:
    def __init__(self, dataset_dir, batch_size=200, read_labels=False,
                 device='cpu', as_tensor=True,  # keep default True since SubDataset already returns tensors
                 contextual_embed=False, plm_model="all-mpnet-base-v2"):

        # Load core data
        self.load_data(dataset_dir, read_labels)
        self.vocab_size = len(self.vocab)

        print("===>train_size: ", self.train_bow.shape[0])
        print("===>test_size: ", self.test_bow.shape[0])
        print("===>vocab_size: ", self.vocab_size)
        print("===>average length: {:.3f}".format(
            self.train_bow.sum(1).sum() / self.train_bow.shape[0]))

        # Paths to sub-embeddings
        train_sub_path = os.path.join(dataset_dir, 'with_sub_bert', 'train_sub_bert.npz')
        test_sub_path  = os.path.join(dataset_dir, 'with_sub_bert', 'test_sub_bert.npz')

        if not os.path.isfile(train_sub_path):
            raise FileNotFoundError(f"Missing sub-embeddings: {train_sub_path}")
        if not os.path.isfile(test_sub_path):
            print("[WARN] Missing test sub-embeddings; using empty list per doc.")
            # You can decide to raise instead:
            # raise FileNotFoundError(f"Missing sub-embeddings: {test_sub_path}")

        # Build datasets (SubDataset internally converts to torch tensors)
        train_dataset = SubDataset(self.train_bow, train_sub_path, device=device)
        if os.path.isfile(test_sub_path):
            test_dataset = SubDataset(self.test_bow, test_sub_path, device=device)
        else:
            # Fallback: no subs for test (each doc gets empty list)
            class EmptySubDataset(SubDataset):
                def __init__(self, bows_np, device="cpu"):
                    self.device = device
                    self.bows = torch.from_numpy(bows_np).to(device)
                    self.E = np.zeros((0, 1), dtype=np.float32)
                    self.indptr = np.arange(self.bows.shape[0] + 1, dtype=np.int64)

                def __getitem__(self, i):
                    return {"data": self.bows[i], "sub_contextual_embed": []}
            test_dataset = EmptySubDataset(self.test_bow, device=device)

        # DataLoaders: stack only 'data'; keep sub lists as-is
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_keep_lists
        )
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_keep_lists
        )
        
        self.train_data = self.train_bow
        self.test_data = self.test_bow

        self.train_data = torch.from_numpy(self.train_data).to(device)
        self.test_data = torch.from_numpy(self.test_data).to(device)
        
        self.train_sub_sent_embeddings = _npz_to_list_of_lists(
            train_sub_path, num_docs=self.train_bow.shape[0], device=device
        )
        self.test_sub_sent_embeddings = _npz_to_list_of_lists(
                test_sub_path, num_docs=self.test_bow.shape[0], device=device
            )

    def load_data(self, path, read_labels):
        self.train_bow = scipy.sparse.load_npz(
            f'{path}/train_bow.npz').toarray().astype('float32')
        self.test_bow = scipy.sparse.load_npz(
            f'{path}/test_bow.npz').toarray().astype('float32')
        self.pretrained_WE = scipy.sparse.load_npz(
            f'{path}/word_embeddings.npz').toarray().astype('float32')

        self.train_texts = file_utils.read_text(f'{path}/train_texts.txt')
        self.test_texts = file_utils.read_text(f'{path}/test_texts.txt')

        if read_labels:
            self.train_labels = np.loadtxt(f'{path}/train_labels.txt', dtype=int)
            self.test_labels = np.loadtxt(f'{path}/test_labels.txt', dtype=int)

        self.vocab = file_utils.read_text(f'{path}/vocab.txt')