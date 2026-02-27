import os
import subprocess
import json

import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC


def _load_dotenv_if_available():
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(env_path):
        load_dotenv(env_path)


def compute_topic_diversity(top_words):
    if not top_words:
        return 0.0
    top_words_text = [" ".join(words) for words in top_words]
    k = len(top_words_text)
    t = len(top_words[0])
    vocab = {}
    for line in top_words_text:
        for w in line.split():
            vocab[w] = vocab.get(w, 0) + 1
    unique_once = sum(1 for v in vocab.values() if v == 1)
    return unique_once / float(k * t)


def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def inverse_purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=1)) / np.sum(contingency_matrix)


def harmonic_purity_score(y_true, y_pred):
    purity = purity_score(y_true, y_pred)
    inverse_purity = inverse_purity_score(y_true, y_pred)
    return (2.0 * purity * inverse_purity) / (purity + inverse_purity + 1e-12)


def evaluate_clustering(theta, labels):
    preds = np.argmax(theta, axis=1)
    return {
        "Purity": purity_score(labels, preds),
        "InversePurity": inverse_purity_score(labels, preds),
        "HarmonicPurity": harmonic_purity_score(labels, preds),
        "NMI": metrics.cluster.normalized_mutual_info_score(labels, preds),
        "ARI": metrics.adjusted_rand_score(labels, preds),
    }


def evaluate_classification(train_theta, test_theta, train_labels, test_labels, tune=False):
    train_labels = np.asarray(train_labels).ravel()
    test_labels = np.asarray(test_labels).ravel()

    if tune:
        best = {"acc": 0.0, "macro-F1": 0.0}
        for c in [0.1, 1, 10, 100, 1000]:
            for gamma in ["scale", "auto", 10, 1, 0.1, 0.01, 0.001]:
                for kernel in ["rbf", "linear"]:
                    clf = SVC(C=c, kernel=kernel, gamma=gamma)
                    clf.fit(train_theta, train_labels)
                    preds = clf.predict(test_theta)
                    best["acc"] = max(best["acc"], accuracy_score(test_labels, preds))
                    best["macro-F1"] = max(best["macro-F1"], f1_score(test_labels, preds, average="macro"))
        return best

    clf = SVC(gamma="scale")
    clf.fit(train_theta, train_labels)
    preds = clf.predict(test_theta)
    return {
        "acc": accuracy_score(test_labels, preds),
        "macro-F1": f1_score(test_labels, preds, average="macro"),
    }


def TC_on_wikipedia_llm_itl(top_word_path, tc_metric="C_V"):
    palmetto_jar_path = "palmetto-0.1.5-exec.jar"
    wikipedia_index_dir = "wikipedia_bd"
    if not os.path.exists(palmetto_jar_path):
        raise FileNotFoundError(f"Missing Palmetto jar: {palmetto_jar_path}")

    cmd = ["java", "-jar", palmetto_jar_path, wikipedia_index_dir, tc_metric, top_word_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Palmetto failed (code {proc.returncode}): {proc.stderr.strip()}"
        )

    scores = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("202"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            scores.append(float(parts[1]))
        except ValueError:
            continue

    if not scores:
        raise RuntimeError("No score parsed from Palmetto output.")
    return scores, float(np.mean(scores))


def _read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def llm_eval(top_words_15, llm_model="gpt-4o", out_dir=None, resume=True):
    if out_dir is None:
        raise ValueError("out_dir is required for llm_eval.")

    scores_path = os.path.join(out_dir, "llm_eval_scores.jsonl")
    if resume and os.path.isfile(scores_path):
        scores = {int(x["k"]): int(x["llm_score"]) for x in _read_jsonl(scores_path)}
    else:
        _load_dotenv_if_available()
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError(
                "OpenAI SDK not available. Install `openai` and set OPENAI_API_KEY."
            ) from exc

        client = OpenAI()
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
        for k, words in enumerate(top_words_15):
            user = (
                "Score topic coherence based on top words.\n"
                "Score 1: unrelated words\n"
                "Score 2: somewhat related with noise\n"
                "Score 3: strongly coherent\n"
                f"Topic index: {k}\n"
                f"Top words: {words}"
            )
            response = client.chat.completions.create(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "score_topic",
                            "description": schema["description"],
                            "parameters": schema["parameters"],
                        },
                    }
                ],
                tool_choice={"type": "function", "function": {"name": "score_topic"}},
                temperature=0,
            )
            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                raise RuntimeError("No tool call returned by LLM.")
            res = json.loads(tool_calls[0].function.arguments)
            if int(res["k"]) != k:
                res["k"] = k
            scores[k] = int(res["llm_score"])

        _write_jsonl(scores_path, [{"k": k, "llm_score": v} for k, v in scores.items()])

    if not scores:
        return [], 0.0
    score_list = [scores[k] for k in sorted(scores)]
    return score_list, float(sum(score_list) / len(score_list))
