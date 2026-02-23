import argparse
import json
import os

from datasethandler import file_utils
from dpo.jsonl_io import read_jsonl
from dpo.llm_client import LLMClient
from dpo.preference_builder import _load_top_words_txt


def _word_dict_list(words, word_to_idx):
    out = []
    for w in words:
        if w in word_to_idx:
            out.append({w: int(word_to_idx[w])})
    return out


def _load_descriptions(run_dir):
    desc_path = os.path.join(run_dir, "topic_descriptions.jsonl")
    if not os.path.isfile(desc_path):
        return {}
    return {int(x["k"]): x["main_meaning"] for x in read_jsonl(desc_path)}


def _load_extra_words(run_dir):
    extra_path = os.path.join(run_dir, "extra_words.jsonl")
    if not os.path.isfile(extra_path):
        return {}
    extra_words = {}
    for x in read_jsonl(extra_path):
        k = int(x["k"])
        extra_words[k] = [list(d.values())[0] for d in x.get("extra_words", [])]
    return extra_words


def _build_prompt(vocab, top_words_15, top_words_25, extra_words, descriptions, k):
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    top15 = top_words_15[k]
    top25 = top_words_25[k]

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
    return user


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--llm_model", type=str, default="gpt-4o")
    parser.add_argument("--topic_index", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = os.path.join(args.dataset_dir, args.dataset)
    vocab_path = os.path.join(dataset_path, "vocab.txt")
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"vocab.txt not found: {vocab_path}")

    vocab = file_utils.read_text(vocab_path)
    top_words_15 = _load_top_words_txt(os.path.join(args.run_dir, "top_words_15.txt"))
    top_words_25 = _load_top_words_txt(os.path.join(args.run_dir, "top_words_25.txt"))
    descriptions = _load_descriptions(args.run_dir)
    extra_words = _load_extra_words(args.run_dir)

    k = args.topic_index
    if k < 0 or k >= len(top_words_15):
        raise ValueError(f"topic_index out of range: {k}")

    system = "You are a topic preference labeling assistant."
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
    user = _build_prompt(vocab, top_words_15, top_words_25, extra_words, descriptions, k)

    out_dir = args.out_dir or args.run_dir
    os.makedirs(out_dir, exist_ok=True)

    llm = LLMClient(model=args.llm_model)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "build_preferences",
                "description": schema.get("description", ""),
                "parameters": schema["parameters"],
            },
        }
    ]
    response = llm.client.chat.completions.create(
        model=llm.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "build_preferences"}},
        temperature=0,
    )

    response_dump = response.model_dump() if hasattr(response, "model_dump") else response
    with open(os.path.join(out_dir, f"llm_response_k{k}.json"), "w", encoding="utf-8") as f:
        json.dump(response_dump, f, ensure_ascii=False, indent=2)

    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        raise RuntimeError("No tool call returned by the model.")

    args_str = tool_calls[0].function.arguments
    with open(os.path.join(out_dir, f"llm_arguments_k{k}.txt"), "w", encoding="utf-8") as f:
        f.write(args_str)

    try:
        parsed = json.loads(args_str)
    except json.JSONDecodeError as exc:
        print("JSONDecodeError:", exc)
        print("Saved raw arguments to:", os.path.join(out_dir, f"llm_arguments_k{k}.txt"))
        raise

    with open(os.path.join(out_dir, f"llm_arguments_k{k}.json"), "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False, indent=2)

    print("OK. Parsed tool arguments saved.")


if __name__ == "__main__":
    main()
