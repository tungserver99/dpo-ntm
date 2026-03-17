# Code for LLMFeed

## Preparing libraries
1. Install the following libraries
    ```
    numpy 
    pytorch 
    sentence_transformers
    scipy 
    gensim
    wandb
    openai
    ```
2. Install java
3. Download and extract [this processed Wikipedia corpus](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/Wikipedia_bd.zip) to ./datasets/wikipedia/ as an external reference corpus.

### Prerequisites

- `OPENAI_API_KEY` set in `.env` or environment
- `sentence-transformers` available (used to embed vocab/description)
- `openai` Python SDK available (used for LLM function calling)

## Basic training

Example:

```bash
python main.py \
  --model ECRTM \
  --dataset 20NG \
  --num_topics 50 \
  --epochs 500 \
  --weight_ECR 120 \
  --use_pretrainWE
```

Important CLI arguments are defined in [utils/config.py]:
- `--dataset`
- `--model` with current choice `ECRTM`
- `--num_topics`
- `--dropout`
- `--weight_ECR`
- `--use_pretrainWE`
- `--train_WE`
- `--epochs`
- `--batch_size`
- `--lr`
- `--device`
- `--lr_scheduler`
- `--lr_step_size`
- `--freeze_we_epoch`
- `--enable_llm_eval`
- `--tune_SVM`

## Update phase: DPO + contrastive refinement

The current update pipeline is controlled by these flags:
- `--enable_update`
- `--update_start_epoch`
- `--update_only`
- `--update_dir`
- `--update_llm_model`
- `--dpo_topic_filter`
- `--dpo_weight`
- `--dpo_alpha`
- `--contrastive_weight`
- `--contrastive_ramp_epochs`
- `--contrastive_topk`
- `--contrastive_temperature`
- `--contrastive_queue_size`
- `--contrastive_doc_encoder`
- `--contrastive_loss_type`
- `--doc_embedding_source`
- `--force_rebuild_doc_embeddings`

### End-to-end update run

Example:

```bash
python main.py \
  --model ECRTM \
  --dataset 20NG \
  --num_topics 50 \
  --epochs 500 \
  --weight_ECR 120 \
  --use_pretrainWE \
  --enable_update \
  --update_start_epoch 350 \
  --update_llm_model gpt-4o \
  --dpo_topic_filter cv_below_avg \
  --dpo_weight 30.0 \
  --dpo_alpha 1.0 \
  --contrastive_weight 20.0 \
  --contrastive_ramp_epochs 0 \
  --contrastive_topk 2 \
  --contrastive_temperature 0.5 \
  --contrastive_queue_size 0 \
  --contrastive_doc_encoder BAAI/bge-small-en-v1.5 \
  --contrastive_loss_type supcon \
  --doc_embedding_source rebuild_from_text \
  --enable_llm_eval
```

What happens during the update phase:
- the trainer saves a snapshot at `update_start_epoch`
- top words for 10, 15, 20, and 25 terms are exported
- topic descriptions and preference data are built with the LLM
- DPO preferences are filtered by the selected topic filter
- document-topic pseudo labels are prepared for contrastive training
- training resumes from the snapshot and continues until `--epochs`

### Reuse existing update artifacts

If you already have update artifacts in a previous run, you can resume only the update stage:

```bash
python main.py \
  --model ECRTM \
  --dataset 20NG \
  --num_topics 50 \
  --epochs 500 \
  --weight_ECR 120 \
  --use_pretrainWE \
  --enable_update \
  --update_only \
  --update_start_epoch 350 \
  --update_dir results/ECRTM/20NG/50/<previous_run>/base_content \
  --update_llm_model gpt-4o \
  --dpo_topic_filter none \
  --dpo_weight 30.0 \
  --dpo_alpha 1.0
```

For `--update_only`, the current code expects at least:
- `preferences.jsonl`
- `top_words_15.txt`
- `beta_ref_logits.npy`
- `update_snapshot_epoch_<E>.pth`

Optional files that will also be reused if present:
- `topic_scores.jsonl`
- `topic_descriptions.jsonl`

## Standalone preference building

You can build the preference artifacts for an existing run directory without rerunning training:

```bash
python dpo_build.py \
  --run_dir results/ECRTM/20NG/50/<run_dir>/base_content \
  --dataset 20NG \
  --plm_model BAAI/bge-small-en-v1.5 \
  --llm_model gpt-4o \
  --device cuda
```

This entrypoint is implemented in [dpo_build.py].

## Run outputs

A normal run directory contains files such as:
- `config.txt`
- `main.log`
- `beta.npy`
- `train_theta.npy`
- `test_theta.npy`
- `train_argmax_theta.npy`
- `test_argmax_theta.npy`
- `top_words_10.txt`
- `top_words_15.txt`
- `top_words_20.txt`
- `top_words_25.txt`
- `checkpoints/checkpoint_epoch_<n>.pth`

When `--enable_update` is active, the run also creates a `base_content/` directory that stores update artifacts such as:
- `update_snapshot_epoch_<E>.pth`
- `beta_ref_logits.npy`
- `preferences.jsonl`
- `topic_scores.jsonl`
- `topic_descriptions.jsonl`
- `extra_words.jsonl`
- `top_words_10.jsonl`
- `top_words_15.jsonl`
- `top_words_20.jsonl`
- `top_words_25.jsonl`
- `update_selected_topics.jsonl`

The final run directory still receives the end-of-training outputs and final evaluation logs.

## Evaluation

[evaluate.py] currently computes:
- topic diversity on top-15 words
- clustering metrics: `NMI`, `Purity`, `InversePurity`, `HarmonicPurity`, `ARI`
- classification metrics: `Accuracy`, `Macro-f1`
- optional LLM topic evaluation
- Wikipedia-based `C_V` and `NPMI` using Palmetto when available

For Wikipedia-based coherence, the code expects:
- Java available on PATH
- `evaluations/palmetto-0.1.5-exec.jar`
- `datasets/wikipedia/wikipedia_bd`

The repository already includes Palmetto jars under `evaluations/`, but the external Wikipedia index still needs to exist if you want those metrics.

## Notes

- `ECRTM` is the only model exposed by the current CLI.
- The LLM model defaults to `gpt-4o` in the current code.

## Acknowledgement
Some part of this implementation is based on [TopMost](https://github.com/BobXWu/TopMost). We also utilizes [Palmetto](https://github.com/dice-group/Palmetto) for the evaluation of topic coherence.

