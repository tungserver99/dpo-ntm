from pathlib import Path
import sys
import types
import os

os.chdir(Path(__file__).resolve().parents[1])

# Light stubs to allow importing ecrtm without full training dependencies.
sys.modules.setdefault("ot", types.SimpleNamespace())
sys.modules.setdefault(
    "transformers",
    types.SimpleNamespace(AutoModelForCausalLM=object, AutoTokenizer=object),
)
from topic_models import ecrtm


def main():
    argv_old = sys.argv
    try:
        sys.argv = [
            "ecrtm.py",
            "--dataset",
            "20News",
            "--epochs",
            "7",
            "--lr",
            "0.123",
            "--batch_size",
            "11",
            "--weight_loss_ECR",
            "9",
            "--llm_step",
            "2",
        ]
        args = ecrtm.parse_args()
    finally:
        sys.argv = argv_old

    assert args.epochs == 7
    assert abs(args.lr - 0.123) < 1e-12
    assert args.batch_size == 11
    assert abs(args.weight_loss_ECR - 9) < 1e-12
    assert args.warmStep == 5
    print("PASS")


if __name__ == "__main__":
    main()
