#!/usr/bin/env bash
set -euo pipefail

TOPICS=(25 50 75 100)
DATASET="YahooAnswers"

for T in "${TOPICS[@]}"; do
  echo "[RUN] dataset=${DATASET} num_topics=${T}"

  python main.py \
    --dataset "${DATASET}" \
    --model ECRTM \
    --num_topics "${T}" \
    --dropout 0.2 \
    --weight_ECR 120.0 \
    --use_pretrainWE \
    --wandb_prj ntm-dpo-con \
    --epochs 500 \
    --batch_size 200 \
    --lr 0.002 \
    --device cuda \
    --seed 0 \
    --lr_scheduler StepLR \
    --lr_step_size 125 \
    --freeze_we_epoch -1 \
    --enable_update \
    --update_start_epoch 350 \
    --update_llm_model gpt-4o \
    --dpo_topic_filter cv_below_avg \
    --dpo_weight 20.0 \
    --dpo_alpha 1.0 \
    --contrastive_weight 10.0 \
    --contrastive_ramp_epochs 0 \
    --contrastive_topk 2 \
    --contrastive_temperature 0.5 \
    --contrastive_queue_size 0 \
    --contrastive_doc_encoder BAAI/bge-small-en-v1.5 \
    --contrastive_loss_type supcon \
    --doc_embedding_source rebuild_from_text \
    --enable_llm_eval

done

echo "All runs finished."