import json
import os
import gc
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from dpo.jsonl_io import read_jsonl


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")


def _load_top_words_txt(path: str) -> List[List[str]]:
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(line.split())
    return rows


def _load_descriptions(run_dir: str, fallback: Optional[Dict[int, str]] = None) -> Dict[int, str]:
    if fallback:
        return {int(k): str(v) for k, v in fallback.items()}
    desc_path = os.path.join(run_dir, "topic_descriptions.jsonl")
    if not os.path.isfile(desc_path):
        return {}
    items = read_jsonl(desc_path)
    out = {}
    for x in items:
        out[int(x["k"])] = str(x.get("main_meaning", ""))
    return out


def _load_train_texts(dataset_dir: str, fallback_texts: Optional[List[str]] = None) -> List[str]:
    texts_path = os.path.join(dataset_dir, "train_texts.txt")
    if os.path.isfile(texts_path):
        with open(texts_path, "r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]

    raw_path = os.path.join(dataset_dir, "train_raw.jsonl")
    if not os.path.isfile(raw_path):
        return fallback_texts if fallback_texts is not None else []

    records: List[Tuple[int, str]] = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            rid = int(item.get("id", i))
            text = str(item.get("text", ""))
            records.append((rid, text))

    records.sort(key=lambda x: x[0])
    return [text for _, text in records]


def _precomputed_doc_embedding_path(dataset_dir: str) -> str:
    return os.path.join(dataset_dir, "with_bgesmall", "train_bgesmall.npz")


def _load_precomputed_doc_embeddings(dataset_dir: str) -> Optional[np.ndarray]:
    primary_path = _precomputed_doc_embedding_path(dataset_dir)
    legacy_path = os.path.join(dataset_dir, "with_bgesmall", "train_raw_bgesmall.npz")
    path = primary_path if os.path.isfile(primary_path) else legacy_path
    if not os.path.isfile(path):
        return None

    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "embeddings" in data.files:
            emb = data["embeddings"]
        elif "arr_0" in data.files:
            emb = data["arr_0"]
        else:
            emb = data[data.files[0]]
    else:
        emb = data
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2:
        raise RuntimeError(f"Invalid precomputed embedding shape at {path}: {emb.shape}")
    return emb


def _save_precomputed_doc_embeddings(dataset_dir: str, emb: np.ndarray) -> str:
    path = _precomputed_doc_embedding_path(dataset_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, embeddings=np.asarray(emb, dtype=np.float32))
    return path


def _build_topic_texts(num_topics: int, descriptions: Dict[int, str], top_words_20: List[List[str]]) -> List[str]:
    topic_texts: List[str] = []
    for k in range(num_topics):
        desc = descriptions.get(k, "").strip()
        words = top_words_20[k] if k < len(top_words_20) else []
        if not desc:
            desc = " ".join(words[:4]).strip()
        topic_texts.append(
            f"Topic about {desc} with top words as: {' '.join(words[:20])}".strip()
        )
    return topic_texts


def _encode_texts(
    texts: List[str],
    model_name: str,
    device: str,
    cache_path: Optional[str],
    batch_size: int = 64,
) -> np.ndarray:
    if cache_path and os.path.isfile(cache_path):
        return np.load(cache_path)
    requested_device = str(device)

    def _encode_once(run_device: str, run_batch_size: int) -> np.ndarray:
        model = SentenceTransformer(model_name, device=run_device)
        emb_local = model.encode(
            texts,
            batch_size=run_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return np.asarray(emb_local, dtype=np.float32)

    def _is_cuda_oom(exc: BaseException) -> bool:
        return "CUDA out of memory" in str(exc)

    try:
        emb = _encode_once(requested_device, batch_size)
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        if not _is_cuda_oom(exc):
            raise
        if not requested_device.startswith("cuda"):
            raise
        torch.cuda.empty_cache()
        gc.collect()
        fallback_bs = max(1, batch_size // 2)
        try:
            emb = _encode_once(requested_device, fallback_bs)
        except (torch.OutOfMemoryError, RuntimeError) as exc2:
            if not _is_cuda_oom(exc2):
                raise
            torch.cuda.empty_cache()
            gc.collect()
            emb = _encode_once("cpu", max(1, fallback_bs // 2))
    if cache_path:
        np.save(cache_path, emb)
    return emb


def _resolve_doc_embeddings(
    dataset_dir: str,
    model_name: str,
    device: str,
    fallback_train_texts: Optional[List[str]],
    doc_embedding_source: str,
    force_rebuild_embeddings: bool,
    logger=None,
) -> np.ndarray:
    precomputed_doc_emb = None
    if doc_embedding_source == "prefer_precomputed" and not force_rebuild_embeddings:
        precomputed_doc_emb = _load_precomputed_doc_embeddings(dataset_dir)
        if precomputed_doc_emb is not None:
            if logger is not None:
                logger.info(f"Loaded precomputed doc embeddings: {_precomputed_doc_embedding_path(dataset_dir)}")
            return precomputed_doc_emb

    train_texts = _load_train_texts(dataset_dir, fallback_texts=fallback_train_texts)
    if len(train_texts) == 0:
        raise RuntimeError("No train texts found for contrastive document embedding build.")
    doc_emb = _encode_texts(train_texts, model_name, device, cache_path=None, batch_size=64)
    saved_path = _save_precomputed_doc_embeddings(dataset_dir, doc_emb)
    if logger is not None:
        logger.info(f"Built doc embeddings from text and saved: {saved_path}")
    return doc_emb


def build_doc_topic_multi_hot(
    run_dir: str,
    dataset_dir: str,
    num_topics: int,
    topk: int,
    model_name: str,
    device: str,
    fallback_descriptions: Optional[Dict[int, str]] = None,
    fallback_train_texts: Optional[List[str]] = None,
    read_only: bool = False,
    force_rebuild_embeddings: bool = False,
    doc_embedding_source: str = "rebuild_from_text",
    logger=None,
) -> np.ndarray:
    model_key = _safe_name(model_name)
    out_path = os.path.join(run_dir, f"doc_topic_topk_{model_key}_k{topk}.npz")
    if os.path.isfile(out_path) and not force_rebuild_embeddings:
        return np.load(out_path)["arr_0"].astype(np.bool_)
    if read_only and force_rebuild_embeddings:
        raise RuntimeError("Cannot force rebuild embeddings in update_only mode.")

    descriptions = _load_descriptions(run_dir, fallback=fallback_descriptions)
    top_words_20_path = os.path.join(run_dir, "top_words_20.txt")
    top_words_20 = _load_top_words_txt(top_words_20_path) if os.path.isfile(top_words_20_path) else []
    topic_texts = _build_topic_texts(num_topics, descriptions, top_words_20)

    topic_emb_path = os.path.join(run_dir, f"topic_desc_embeddings_{model_key}.npy")
    topic_cache_path = None if force_rebuild_embeddings else topic_emb_path
    topic_emb = _encode_texts(topic_texts, model_name, device, topic_cache_path, batch_size=64)
    if force_rebuild_embeddings:
        np.save(topic_emb_path, topic_emb)
        if logger is not None:
            logger.info(f"Rebuilt topic description embeddings: {topic_emb_path}")

    doc_emb = _resolve_doc_embeddings(
        dataset_dir=dataset_dir,
        model_name=model_name,
        device=device,
        fallback_train_texts=fallback_train_texts,
        doc_embedding_source=doc_embedding_source,
        force_rebuild_embeddings=force_rebuild_embeddings,
        logger=logger,
    )

    topic_norm = topic_emb / (np.linalg.norm(topic_emb, axis=1, keepdims=True) + 1e-12)
    doc_norm = doc_emb / (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-12)
    sims = np.matmul(doc_norm, topic_norm.T)

    k = max(1, min(int(topk), int(num_topics)))
    topk_idx = np.argpartition(sims, -k, axis=1)[:, -k:]

    multi_hot = np.zeros((doc_norm.shape[0], num_topics), dtype=np.bool_)
    rows = np.arange(doc_norm.shape[0])[:, None]
    multi_hot[rows, topk_idx] = True

    np.savez_compressed(out_path, multi_hot)
    if logger is not None:
        logger.info(f"Contrastive doc-topic assignments saved: {out_path}")
    return multi_hot


def supcon_theta_loss(
    theta: torch.Tensor,
    topic_multi_hot: torch.Tensor,
    queue_theta: Optional[torch.Tensor],
    queue_topic_multi_hot: Optional[torch.Tensor],
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, int]:
    if theta.ndim != 2 or topic_multi_hot.ndim != 2:
        raise ValueError("theta and topic_multi_hot must be 2D tensors.")
    if theta.shape[0] == 0:
        return torch.tensor(0.0, device=theta.device), 0

    z = F.normalize(theta, p=2, dim=1)
    all_z = z
    all_topics = topic_multi_hot
    batch_size = z.shape[0]

    if queue_theta is not None and queue_theta.numel() > 0:
        all_z = torch.cat([all_z, F.normalize(queue_theta, p=2, dim=1)], dim=0)
        all_topics = torch.cat([all_topics, queue_topic_multi_hot], dim=0)

    overlap = (topic_multi_hot.float() @ all_topics.float().T) > 0
    logits = torch.matmul(z, all_z.T) / max(float(temperature), 1e-6)

    self_mask = torch.zeros_like(overlap, dtype=torch.bool)
    self_mask[:, :batch_size] = torch.eye(batch_size, dtype=torch.bool, device=theta.device)

    valid_mask = ~self_mask
    pos_mask = overlap & valid_mask

    if not pos_mask.any():
        return torch.tensor(0.0, device=theta.device), 0

    logits = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(logits) * valid_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    pos_count = pos_mask.sum(dim=1)
    anchor_mask = pos_count > 0
    mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / (pos_count.float() + 1e-12)
    loss = -mean_log_prob_pos[anchor_mask].mean()
    return loss, int(anchor_mask.sum().item())


def infonce_theta_loss(
    theta: torch.Tensor,
    topic_multi_hot: torch.Tensor,
    queue_theta: Optional[torch.Tensor],
    queue_topic_multi_hot: Optional[torch.Tensor],
    temperature: float = 0.07,
) -> Tuple[torch.Tensor, int]:
    if theta.ndim != 2 or topic_multi_hot.ndim != 2:
        raise ValueError("theta and topic_multi_hot must be 2D tensors.")
    if theta.shape[0] == 0:
        return torch.tensor(0.0, device=theta.device), 0

    z = F.normalize(theta, p=2, dim=1)
    all_z = z
    all_topics = topic_multi_hot
    batch_size = z.shape[0]

    if queue_theta is not None and queue_theta.numel() > 0:
        all_z = torch.cat([all_z, F.normalize(queue_theta, p=2, dim=1)], dim=0)
        all_topics = torch.cat([all_topics, queue_topic_multi_hot], dim=0)

    anchor_dot_contrast = torch.matmul(z, all_z.T) / max(float(temperature), 1e-6)
    logits_max = anchor_dot_contrast.max(dim=1, keepdim=True).values
    logits = anchor_dot_contrast - logits_max.detach()

    overlap = (topic_multi_hot.float() @ all_topics.float().T) > 0
    pair_pos_mask = overlap.float()
    logits_mask = torch.ones_like(pair_pos_mask)
    logits_mask[:, :batch_size] = logits_mask[:, :batch_size] - torch.eye(
        batch_size, device=theta.device, dtype=logits_mask.dtype
    )
    pair_pos_mask = pair_pos_mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask * (1.0 - pair_pos_mask)
    sum_exp_logits = exp_logits.sum(dim=1, keepdim=True)

    pos_count = pair_pos_mask.sum(dim=1)
    valid = (pos_count > 0) & (sum_exp_logits.squeeze(1) > 0)
    if not valid.any():
        return torch.tensor(0.0, device=theta.device), 0

    log_prob = logits * logits_mask - torch.log(sum_exp_logits + 1e-10)
    mean_log_prob_pos = (pair_pos_mask * log_prob).sum(dim=1) / pos_count.clamp_min(1.0)
    loss = -mean_log_prob_pos[valid].mean()
    return loss, int(valid.sum().item())
