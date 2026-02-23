import argparse
import os

from datasethandler import file_utils
from dpo.preference_builder import build_preference_pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="existing run directory containing top_words_*.txt")
    parser.add_argument("--dataset", type=str, required=True,
                        help="dataset name under ./datasets")
    parser.add_argument("--dataset_dir", type=str, default="datasets",
                        help="base dataset directory")
    parser.add_argument("--plm_model", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--llm_model", type=str, default="gpt-4o")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = os.path.join(args.dataset_dir, args.dataset)
    vocab_path = os.path.join(dataset_path, "vocab.txt")
    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"vocab.txt not found: {vocab_path}")

    vocab = file_utils.read_text(vocab_path)

    build_preference_pipeline(
        run_dir=args.run_dir,
        vocab=vocab,
        plm_model=args.plm_model,
        llm_model=args.llm_model,
        device=args.device,
        resume=args.resume,
        only_preferences=False,
    )

    print(f"[DPO] Preference build completed for run_dir: {args.run_dir}")


if __name__ == "__main__":
    main()
