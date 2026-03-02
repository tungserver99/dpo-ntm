@echo off
setlocal EnableDelayedExpansion

set "DATASET=SearchSnippets"
set "TOPICS=25"
set "WEIGHT_ECR=50.0"
set "ARTIFACT_DIR=results\ECRTM\SearchSnippets\25\50.0-500-2026-02-25_23-50-45\base_content"

set "DPO_WEIGHTS=40 20 30 50 15"
set "DPO_ALPHAS=1.0 1.5 2.0 3.0 0.5"
set "CONTRASTIVE_WEIGHTS=5 10 15 20"
set "TOPIC_FILTERS=none cv_below_avg either llm_score_1_2"

for %%F in (%TOPIC_FILTERS%) do (
  for %%W in (%DPO_WEIGHTS%) do (
    for %%A in (%DPO_ALPHAS%) do (
      for %%C in (%CONTRASTIVE_WEIGHTS%) do (
        echo [RUN] filter=%%F dpo_weight=%%W dpo_alpha=%%A contrastive_weight=%%C

        python main.py ^
          --dataset %DATASET% ^
          --model ECRTM ^
          --num_topics %TOPICS% ^
          --dropout 0.2 ^
          --weight_ECR %WEIGHT_ECR% ^
          --use_pretrainWE ^
          --wandb_prj ntm-dpo-con ^
          --epochs 500 ^
          --batch_size 200 ^
          --lr 0.002 ^
          --device cuda ^
          --seed 0 ^
          --lr_scheduler StepLR ^
          --lr_step_size 125 ^
          --enable_update ^
          --update_only ^
          --update_dir "%ARTIFACT_DIR%" ^
          --update_start_epoch 350 ^
          --update_llm_model gpt-4o ^
          --dpo_topic_filter %%F ^
          --dpo_weight %%W ^
          --dpo_alpha %%A ^
          --contrastive_weight %%C ^
          --contrastive_ramp_epochs 0 ^
          --contrastive_topk 2 ^
          --contrastive_temperature 0.5 ^
          --contrastive_queue_size 0 ^
          --contrastive_doc_encoder BAAI/bge-small-en-v1.5 ^
          --contrastive_loss_type supcon ^
          --doc_embedding_source rebuild_from_text ^
          --enable_llm_eval

        if errorlevel 1 (
          echo [ERROR] Failed at filter=%%F dpo_weight=%%W dpo_alpha=%%A contrastive_weight=%%C
          exit /b 1
        )
      )
    )
  )
)

echo Grid search completed.
endlocal
