import argparse
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='ECRTM')
parser.add_argument('--dataset', type=str, default='20News')
parser.add_argument('--n_topic', type=int, default=50)
parser.add_argument('--seed', '--random_seed', dest='seed', type=int, default=1)
parser.add_argument('--eval_step', type=int, default=50)
parser.add_argument('--inference_bs', type=int, default=100) # set this number based on your GPU memory
parser.add_argument('--llm_step', type=int, default=50) # the number of epochs for llm refine
parser.add_argument('--eval_topics', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--enable_llm_eval', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--llm_model', type=str, default='gpt-4o')
args, passthrough = parser.parse_known_args()


if __name__ == '__main__':
    train_command = [
        sys.executable,
        "-m",
        "topic_models.ecrtm",
        f"--name={args.name}",
        f"--dataset={args.dataset}",
        f"--n_topic={args.n_topic}",
        f"--seed={args.seed}",
        f"--eval_step={args.eval_step}",
        f"--inference_bs={args.inference_bs}",
        f"--llm_step={args.llm_step}",
        "--llm_itl",
    ] + passthrough
    train_proc = subprocess.run(train_command, check=False)
    if train_proc.returncode != 0:
        raise SystemExit(train_proc.returncode)

    model_folder = f"{args.name}_{args.dataset}_K{args.n_topic}_seed{args.seed}_useLLM-True"
    eval_command = [
        sys.executable,
        "-m",
        "evaluation.eval_ecrtm",
        f"--model_folder={model_folder}",
        f"--dataset={args.dataset}",
        f"--inference_bs={args.inference_bs}",
    ]
    if args.eval_topics:
        eval_command.append("--eval_topics")
    if args.enable_llm_eval:
        eval_command.append("--enable_llm_eval")
        eval_command.append(f"--llm_model={args.llm_model}")
    subprocess.run(eval_command, check=False)
