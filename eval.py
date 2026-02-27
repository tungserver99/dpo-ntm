import argparse
import os
import subprocess
import sys


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='20News')
parser.add_argument('--n_topic', type=int, default=50)
parser.add_argument('--eval_topics', action='store_true')
args = parser.parse_args()


if __name__ == '__main__':
    model_folders = os.listdir('save_models')
    model_folders = [f for f in model_folders if ('%s_%s_K%s' %
                     ('ecrtm', args.dataset, args.n_topic)).lower() in f.lower()
                     and 'useLLM-True'.lower() in f.lower()]
    print('Evaluation for:' )
    print(model_folders)

    for model_folder in model_folders:
        command = [
            sys.executable,
            "-m",
            "evaluation.eval_ecrtm",
            f"--model_folder={model_folder}",
            f"--dataset={args.dataset}",
        ]
        if args.eval_topics:
            command.append("--eval_topics")
        subprocess.run(command, check=False)


