import os

from dpo.jsonl_io import read_jsonl, write_jsonl
from dpo.llm_client import LLMClient


def llm_eval(top_words_15, llm_model="gpt-4o", out_dir=None, resume=True):
    if out_dir is None:
        raise ValueError("out_dir is required for llm_eval.")

    scores_path = os.path.join(out_dir, "llm_eval_scores.jsonl")
    if resume and os.path.isfile(scores_path):
        scores = {int(x["k"]): int(x["llm_score"]) for x in read_jsonl(scores_path)}
    else:
        llm = LLMClient(model=llm_model)
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
        scores = {}
        for k, words in enumerate(top_words_15):
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

        write_jsonl(scores_path, [{"k": k, "llm_score": v} for k, v in scores.items()])

    if not scores:
        return [], 0.0
    score_list = [scores[k] for k in sorted(scores)]
    mean_score = sum(score_list) / float(len(score_list))
    return score_list, mean_score
