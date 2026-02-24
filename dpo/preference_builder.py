import json
import os
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from dpo.jsonl_io import read_jsonl, write_jsonl
from dpo.llm_client import LLMClient


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")


def _load_top_words_txt(path: str) -> List[List[str]]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(line.split())
    return lines


def _word_dict_list(words: List[str], word_to_idx: Dict[str, int]) -> List[Dict[str, int]]:
    out = []
    for w in words:
        if w in word_to_idx:
            out.append({w: int(word_to_idx[w])})
    return out


def _save_top_words_jsonl(
    out_path: str,
    top_words: List[List[str]],
    word_to_idx: Dict[str, int],
    scores: Dict[int, int],
    descriptions: Dict[int, str],
):
    items = []
    for k, words in enumerate(top_words):
        items.append(
            {
                "k": k,
                "main meaning": descriptions.get(k, ""),
                "llm_score": scores.get(k, ""),
                "top_words": _word_dict_list(words, word_to_idx),
            }
        )
    write_jsonl(out_path, items)


def _score_topics(llm: LLMClient, top_words_15: List[List[str]]):
    scores = {}
    schema = {
        "description": "Score topic coherence from top words",
        "parameters": {
            "type": "object",
            "properties": {
                "k": {"type": "integer"},
                "llm_score": {"type": "integer", "enum": [1, 2, 3]},
            },
            "required": ["k", "llm_score"],
        },
    }
    system = "You are a topic modeling evaluator."
    for k, words in enumerate(tqdm(top_words_15, desc="DPO scoring topics")):
        user = (
            "Score topic coherence based on top words.\n"
            "Score 1: unrelated words\n"
            "Score 2: somewhat related with noise\n"
            "Score 3: strongly coherent\n"
            f"Topic index: {k}\n"
            f"Top words: {words}"
        )
        res = llm.call_function(system, user, "score_topic", schema)
        if int(res["k"]) != k:
            res["k"] = k
        scores[k] = int(res["llm_score"])
    return scores


def _describe_topics(llm: LLMClient, top_words_20: List[List[str]], scores: Dict[int, int]):
    descriptions = {}
    schema = {
        "description": "Describe the main meaning of a topic",
        "parameters": {
            "type": "object",
            "properties": {
                "k": {"type": "integer"},
                "main_meaning": {"type": "string"},
            },
            "required": ["k", "main_meaning"],
        },
    }
    system = "You are a topic modeling assistant."
    used = []
    for k, words in enumerate(tqdm(top_words_20, desc="DPO describing topics")):
        score = scores.get(k, 2)
        if score == 1:
            user = (
                "Generate a short descriptive phrase for the topic based on top words. "
                "Use the dominant semantic theme from the top-20 words, prioritizing higher-ranked words. "
                "Prefer a concrete noun phrase (2-6 words) that could label the topic. "
                "Avoid overlapping with existing topic descriptions; choose a distinct semantic theme "
                "and avoid reusing the same head nouns.\n"
                f"Existing topic descriptions: {used}\n"
                f"Topic index: {k}\n"
                f"Top words (20): {words}"
            )
        else:
            user = (
                "Generate a short descriptive phrase for the topic based on top words. "
                "Use the dominant semantic theme from the top-20 words, prioritizing higher-ranked words. "
                "Prefer a concrete noun phrase (2-6 words) that could label the topic.\n"
                f"Topic index: {k}\n"
                f"Top words (20): {words}"
            )
        res = llm.call_function(system, user, "describe_topic", schema)
        if int(res["k"]) != k:
            res["k"] = k
        desc = res["main_meaning"]
        descriptions[k] = desc
        used.append(desc)
    return descriptions


def _embed_vocab(vocab: List[str], model_name: str, device: str, cache_path: str):
    if os.path.isfile(cache_path):
        return np.load(cache_path)
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(vocab, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)
    np.save(cache_path, emb)
    return emb


def _select_extra_words(
    vocab: List[str],
    vocab_emb: np.ndarray,
    descriptions: Dict[int, str],
    top_words_25: List[List[str]],
    model_name: str,
    device: str,
):
    model = SentenceTransformer(model_name, device=device)
    desc_list = [descriptions[k] for k in range(len(top_words_25))]
    desc_emb = model.encode(desc_list, show_progress_bar=True)
    desc_emb = np.asarray(desc_emb, dtype=np.float32)
    # normalize
    vocab_norm = vocab_emb / (np.linalg.norm(vocab_emb, axis=1, keepdims=True) + 1e-12)
    desc_norm = desc_emb / (np.linalg.norm(desc_emb, axis=1, keepdims=True) + 1e-12)

    word_to_idx = {w: i for i, w in enumerate(vocab)}
    extra = {}
    for k, top_words in enumerate(top_words_25):
        top_idx = {word_to_idx[w] for w in top_words if w in word_to_idx}
        sims = np.dot(vocab_norm, desc_norm[k])
        # exclude top-25
        sims[list(top_idx)] = -1e9
        top5 = sims.argsort()[-5:][::-1]
        extra[k] = [int(i) for i in top5]
    return extra


def _build_preferences(
    llm: LLMClient,
    vocab: List[str],
    top_words_15: List[List[str]],
    top_words_25: List[List[str]],
    extra_words: Dict[int, List[int]],
    descriptions: Dict[int, str],
):
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    prefs = {}

    def _normalize_indices(value):
        """Accept int/list/tuple/set/str and normalize to a flat list[int]."""
        if value is None:
            return []
        if isinstance(value, (int, np.integer)):
            return [int(value)]
        if isinstance(value, str):
            parts = [p.strip() for p in value.replace(";", ",").split(",")]
            out = []
            for p in parts:
                if p == "":
                    continue
                try:
                    out.append(int(p))
                except Exception:
                    continue
            return out
        if isinstance(value, (list, tuple, set)):
            out = []
            for x in value:
                if isinstance(x, (int, np.integer)):
                    out.append(int(x))
                else:
                    try:
                        out.append(int(x))
                    except Exception:
                        continue
            return out
        try:
            return [int(value)]
        except Exception:
            return []
    schema = {
        "description": "Identify win/lose word indices for a topic",
        "parameters": {
            "type": "object",
            "properties": {
                "k": {"type": "integer"},
                "w_win_indices": {"type": "array", "items": {"type": "integer"}},
                "w_loose_indices": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["k", "w_win_indices", "w_loose_indices"],
        },
    }
    system = "You are a topic preference labeling assistant."

    for k in tqdm(range(len(top_words_15)), desc="DPO building preferences"):
        top15 = top_words_15[k]
        top25 = top_words_25[k]
        # words ranked 11-25 (15 words) + 5 extra
        other_words = top25[10:25]
        extra_idx = extra_words.get(k, [])
        extra_words_list = [vocab[i] for i in extra_idx]
        other_words = other_words + extra_words_list

        top15_kv = _word_dict_list(top15, word_to_idx)
        other_kv = _word_dict_list(other_words, word_to_idx)

        user = (
            "Identify win/lose word indices for the topic.\n"
            "IMPORTANT: The indices you return must be the vocabulary indices provided in the word:index pairs below.\n"
            "Do NOT use the position in the list. Only use the numeric values shown after each word.\n"
            "Goal: keep good words in top-15, remove bad words in top-15, and promote good words outside top-15.\n"
            "Definition:\n"
            "- Good (win): semantically central and clearly related to the topic description.\n"
            "- Bad (lose): unrelated, off-topic, too generic, or misleading for the topic.\n"
            "Example:\n"
            "If the topic is about 'computer hardware' and the top-15 contains "
            "['cpu','motherboard','bios','floppy','drive','sale','discount','today',...], "
            "good words in top-15 like 'cpu','motherboard' should be win; "
            "bad words inside top-15 like 'sale','discount','today' should be lose; "
            "good words outside top-15 (e.g., 'chipset') should be win.\n"
            "Index example: if you see [{'cpu': 120}, {'gpu': 450}], then valid indices are 120 and 450 (not 0 or 1).\n"
            "Rules:\n"
            "1) Any bad words in top15 must be in lose list.\n"
            "2) Any good words outside top15 must be in win list.\n"
            "3) Good words inside top15 should be kept as win (unless clearly bad).\n"
            "4) Aim for a roughly balanced set; if possible keep win/lose counts close (difference <= 2).\n"
            f"Topic index: {k}\n"
            f"Topic description: {descriptions.get(k, '')}\n"
            f"Top-15 words: {top15_kv}\n"
            f"Other words (rank 11-25 + extra): {other_kv}"
        )
        res = llm.call_function(system, user, "build_preferences", schema)
        if int(res["k"]) != k:
            res["k"] = k

        allowed = {d[list(d.keys())[0]] for d in top15_kv + other_kv}
        w_win_raw = _normalize_indices(res.get("w_win_indices", []))
        w_loose_raw = _normalize_indices(res.get("w_loose_indices", []))
        w_win = [int(i) for i in w_win_raw if int(i) in allowed]
        w_loose = [int(i) for i in w_loose_raw if int(i) in allowed]
        prefs[k] = {"w_win": w_win, "w_loose": w_loose}

    return prefs


def build_preference_pipeline(
    run_dir: str,
    vocab: List[str],
    plm_model: str,
    llm_model: str,
    device: str,
    resume: bool = True,
    only_preferences: bool = False,
):
    if only_preferences:
        raise RuntimeError("only_preferences should be handled in trainer; not in preference_builder.")
    top10_path = os.path.join(run_dir, "top_words_10.txt")
    top15_path = os.path.join(run_dir, "top_words_15.txt")
    top20_path = os.path.join(run_dir, "top_words_20.txt")
    top25_path = os.path.join(run_dir, "top_words_25.txt")

    top_words_10 = _load_top_words_txt(top10_path)
    top_words_15 = _load_top_words_txt(top15_path)
    top_words_20 = _load_top_words_txt(top20_path)
    top_words_25 = _load_top_words_txt(top25_path)

    word_to_idx = {w: i for i, w in enumerate(vocab)}

    llm = LLMClient(model=llm_model)

    scores_path = os.path.join(run_dir, "topic_scores.jsonl")
    desc_path = os.path.join(run_dir, "topic_descriptions.jsonl")
    if resume and os.path.isfile(scores_path):
        scores = {int(x["k"]): int(x["llm_score"]) for x in read_jsonl(scores_path)}
    elif not only_preferences:
        scores = _score_topics(llm, top_words_15)
        write_jsonl(scores_path, [{"k": k, "llm_score": v} for k, v in scores.items()])
    else:
        raise RuntimeError("Missing topic_scores.jsonl while only_preferences is set.")

    if resume and os.path.isfile(desc_path):
        descriptions = {int(x["k"]): x["main_meaning"] for x in read_jsonl(desc_path)}
    elif not only_preferences:
        descriptions = _describe_topics(llm, top_words_20, scores)
        write_jsonl(desc_path, [{"k": k, "main_meaning": v} for k, v in descriptions.items()])
    else:
        raise RuntimeError("Missing topic_descriptions.jsonl while only_preferences is set.")

    # Save top words jsonl for 10/15/20/25
    _save_top_words_jsonl(os.path.join(run_dir, "top_words_10.jsonl"), top_words_10, word_to_idx, scores, descriptions)
    _save_top_words_jsonl(os.path.join(run_dir, "top_words_15.jsonl"), top_words_15, word_to_idx, scores, descriptions)
    _save_top_words_jsonl(os.path.join(run_dir, "top_words_20.jsonl"), top_words_20, word_to_idx, scores, descriptions)
    _save_top_words_jsonl(os.path.join(run_dir, "top_words_25.jsonl"), top_words_25, word_to_idx, scores, descriptions)

    # Embedding-based extra words
    extra_path = os.path.join(run_dir, "extra_words.jsonl")
    if resume and os.path.isfile(extra_path):
        extra_words = {
            int(x["k"]): [list(d.values())[0] for d in x["extra_words"]]
            for x in read_jsonl(extra_path)
        }
    elif not only_preferences:
        model_key = _safe_name(plm_model)
        vocab_emb_cache = os.path.join(run_dir, f"vocab_embeddings_{model_key}.npy")
        vocab_emb = _embed_vocab(vocab, plm_model, device, vocab_emb_cache)
        extra_words = _select_extra_words(vocab, vocab_emb, descriptions, top_words_25, plm_model, device)
        write_jsonl(
            extra_path,
            [
                {"k": k, "extra_words": _word_dict_list([vocab[i] for i in idxs], word_to_idx)}
                for k, idxs in extra_words.items()
            ],
        )
    else:
        raise RuntimeError("Missing extra_words.jsonl while only_preferences is set.")

    # Preferences
    prefs_path = os.path.join(run_dir, "preferences.jsonl")
    if resume and os.path.isfile(prefs_path):
        prefs_list = read_jsonl(prefs_path)
        prefs = {}
        for x in prefs_list:
            k = int(x["k"])
            if "w_win_indices" in x and "w_loose_indices" in x:
                prefs[k] = {"w_win": x["w_win_indices"], "w_loose": x["w_loose_indices"]}
            else:
                prefs[k] = {
                    "w_win": x.get("w_plus_indices", []),
                    "w_loose": x.get("w_minus_indices", []),
                }
    else:
        prefs = _build_preferences(llm, vocab, top_words_15, top_words_25, extra_words, descriptions)
        write_jsonl(
            prefs_path,
            [
                {"k": k, "w_win_indices": v["w_win"], "w_loose_indices": v["w_loose"]}
                for k, v in prefs.items()
            ],
        )

    return {
        "scores": scores,
        "descriptions": descriptions,
        "preferences": prefs,
    }
