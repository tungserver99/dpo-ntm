import argparse
import json
import os

import torch
from tqdm import tqdm

from topic_models.ECRTM.utils.data.TextData import TextData
from evaluation import ntm_eval_adapter as ntm_eval


def _load_topics(topic_file):
    with open(topic_file, "r", encoding="utf-8") as f:
        return [line.strip().split() for line in f if line.strip()]


def _top_words_for_td(topics, topk=15):
    if not topics:
        return topics
    min_len = min(len(t) for t in topics)
    cut = min(topk, min_len)
    return [t[:cut] for t in topics]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument("--model_folder", type=str, required=True)
    parser.add_argument('--dataset', type=str, default='20News')
    parser.add_argument("--inference_bs", type=int, default=500)
    parser.add_argument("--eval_topics", action='store_true')
    parser.add_argument("--tune_svm", action="store_true")
    parser.add_argument("--enable_llm_eval", action="store_true")
    parser.add_argument("--llm_model", type=str, default="gpt-4o")
    args = parser.parse_args()

    checkpoint_folder = 'save_models'
    model_epochs = os.listdir(os.path.join(checkpoint_folder, args.model_folder))
    model_epochs = sorted(model_epochs, key=lambda x: int(x.split('-')[1].split('.')[0]))

    dataset_handler = TextData(args.dataset, args.inference_bs)
    train_data = dataset_handler.train_data
    test_data = dataset_handler.test_data
    train_label = dataset_handler.train_labels
    test_label = dataset_handler.test_labels

    save_path = 'evaluation_output/%s' % (args.model_folder + '.jsonl')
    open(save_path, 'w').close()

    for model_epoch in tqdm(model_epochs):
        model = torch.load(os.path.join(checkpoint_folder, args.model_folder, model_epoch), weights_only=False)
        model.eval()

        with torch.no_grad():
            test_theta = model.get_theta(test_data).cpu().numpy()
            train_theta = model.get_theta(train_data).cpu().numpy()

        metrics = {}
        clustering_results = ntm_eval.evaluate_clustering(test_theta, test_label)
        metrics.update({k: float(v) for k, v in clustering_results.items()})

        classification_results = ntm_eval.evaluate_classification(
            train_theta, test_theta, train_label, test_label, tune=args.tune_svm
        )
        metrics["Accuracy"] = float(classification_results["acc"])
        metrics["Macro-f1"] = float(classification_results["macro-F1"])

        if args.eval_topics:
            epoch_id = model_epoch.split('.')[0].split('-')[1]
            topic_file = 'save_topics/%s/epoch%s_tm_words.txt' % (args.model_folder, epoch_id)
            topics = _load_topics(topic_file)

            td_topics = _top_words_for_td(topics, topk=15)
            metrics["TD_15"] = float(ntm_eval.compute_topic_diversity(td_topics))

            try:
                _, tc_mean = ntm_eval.TC_on_wikipedia_llm_itl(topic_file, tc_metric="C_V")
                metrics["TC_wikipedia_llm_itl"] = float(tc_mean)
            except Exception as exc:
                metrics["TC_wikipedia_llm_itl_error"] = str(exc)

            try:
                _, npmi_mean = ntm_eval.TC_on_wikipedia_llm_itl(topic_file, tc_metric="NPMI")
                metrics["NPMI_wiki"] = float(npmi_mean)
            except Exception as exc:
                metrics["NPMI_wiki_error"] = str(exc)

            if args.enable_llm_eval:
                run_dir = os.path.join("save_topics", args.model_folder)
                try:
                    llm_scores, llm_mean = ntm_eval.llm_eval(
                        td_topics, llm_model=args.llm_model, out_dir=run_dir, resume=True
                    )
                    metrics["LLM_eval_mean"] = float(llm_mean)
                    metrics["LLM_eval_scores"] = llm_scores
                except Exception as exc:
                    metrics["LLM_eval_error"] = str(exc)

        epoch_id = int(model_epoch.split('.')[0].split('-')[1])
        metrics["epoch"] = epoch_id
        with open(save_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(metrics) + '\n')


