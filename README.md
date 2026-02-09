# Code for NeuroMax: Enhancing Neural Topic Modeling via Maximizing Mutual Information and Group Topic Regularization (EMNLP 2024 Findings)

[Paper link](https://arxiv.org/abs/2409.19749)

## Preparing libraries
1. Install the following libraries
    ```
    numpy 1.26.4
    torch_kmeans 0.2.0
    pytorch 2.2.0
    sentence_transformers 2.2.2
    scipy 1.10
    bertopic 0.16.0
    gensim 4.2.0
    ```
2. Install java
3. Download [this java jar](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/palmetto-0.1.0-jar-with-dependencies.jar) to ./evaluations/pametto.jar for evaluating
4. Download and extract [this processed Wikipedia corpus](https://hobbitdata.informatik.uni-leipzig.de/homes/mroeder/palmetto/Wikipedia_bd.zip) to ./datasets/wikipedia/ as an external reference corpus.

## Usage
To run and evaluate our model for YahooAnswers dataset, run this example:

> python main.py --use_pretrainWE

## DPO End-to-End Training (ECRTM)

This repo supports LLM-guided preference learning (DPO) for **ECRTM** in a single run:
training normally → build preference dataset (LLM + embeddings) → DPO fine-tuning.

### Prerequisites

- `OPENAI_API_KEY` set in `.env` or environment
- `sentence-transformers` available (used to embed vocab/description)
- `openai` Python SDK available (used for LLM function calling)

### Key Arguments

- `--enable_dpo` : enable DPO pipeline
- `--dpo_start_epoch` : epoch E to snapshot and start preference building
- `--dpo_weight` : weight for DPO loss
- `--dpo_alpha` : temperature for DPO loss
- `--dpo_topic_filter` : `cv_below_avg` (default) | `llm_score_1_2` | `either`
- `--dpo_llm_model` : LLM model (default `gpt-4o`)
- `--dpo_only_preferences` : skip preference building; use existing `preferences.jsonl`
- `--dpo_run_dir` : run directory containing DPO artifacts to reuse

### Example (End-to-End)

```bash
python main.py \
  --model ECRTM \
  --dataset 20NG \
  --num_topics 50 \
  --epochs 500 \
  --weight_ECR 200 \
  --use_pretrainWE \
  --enable_dpo \
  --dpo_start_epoch 400 \
  --dpo_weight 10.0 \
  --dpo_alpha 1.0 \
  --dpo_topic_filter cv_below_avg \
  --dpo_llm_model gpt-4o
```

### What Gets Saved (per run directory)

- `top_words_10/15/20/25.txt`
- `top_words_10/15/20/25.jsonl`
- `topic_scores.jsonl`
- `topic_descriptions.jsonl`
- `extra_words.jsonl`
- `preferences.jsonl`
- `beta_ref_logits.npy`
- `dpo_snapshot_epoch_E.pth`
- `dpo_selected_topics.jsonl`

### Notes

- DPO uses **beta logits** (pre-softmax) and a **frozen reference** at epoch E.
- If you rerun, the pipeline reuses existing JSONL artifacts to save cost.

### Reuse Existing Preferences (Only DPO Training)

If you want to **skip preference building** and only train with DPO loss, make sure this file exists:
- `preferences.jsonl`

Then run with:

```bash
python main.py \
  --model ECRTM \
  --dataset 20NG \
  --enable_dpo \
  --dpo_only_preferences \
  --dpo_run_dir <path_to_previous_run_dir>
```

### Build Preferences Only (Standalone)

If you want to **build preferences in a specific run directory** without training:

```bash
python dpo_build.py \
  --run_dir <path_to_existing_run_dir> \
  --dataset 20NG \
  --plm_model all-mpnet-base-v2 \
  --llm_model gpt-4o
```

## Acknowledgement
Some part of this implementation is based on [TopMost](https://github.com/BobXWu/TopMost). We also utilizes [Palmetto](https://github.com/dice-group/Palmetto) for the evaluation of topic coherence.

## Citation

If you want to reuse our code, please cite us as:

```
@misc{pham2024neuromax,
      title={NeuroMax: Enhancing Neural Topic Modeling via Maximizing Mutual Information and Group Topic Regularization}, 
      author={Duy-Tung Pham and Thien Trang Nguyen Vu and Tung Nguyen and Linh Ngo Van and Duc Anh Nguyen and Thien Huu Nguyen},
      year={2024},
      eprint={2409.19749},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.19749}, 
}
```
