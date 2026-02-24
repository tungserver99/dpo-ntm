@echo off
setlocal EnableDelayedExpansion

set "TOPICS=25 50 75 100"
set "DATASETS=20NG AGNews BBC_new IMDB NYT WOS_vocab_50K WOS_vocab_5k YahooAnswers"

for %%D in (%DATASETS%) do (
  for %%T in (%TOPICS%) do (
    set "SKIP=0"
    if /I "%%D"=="20NG" if "%%T"=="25" set "SKIP=1"
    if /I "%%D"=="20NG" if "%%T"=="50" set "SKIP=1"

    if "!SKIP!"=="1" (
      echo [SKIP] dataset=%%D num_topics=%%T (already run)
    ) else (
      echo [RUN] dataset=%%D num_topics=%%T
      python main.py ^
        --dataset %%D ^
        --model ECRTM ^
        --num_topics %%T ^
        --dropout 0.2 ^
        --weight_ECR 120.0 ^
        --use_pretrainWE ^
        --wandb_prj ECRTM_TM ^
        --epochs 500 ^
        --batch_size 200 ^
        --lr 0.002 ^
        --device cuda ^
        --seed 0 ^
        --lr_scheduler StepLR ^
        --lr_step_size 125 ^
        --freeze_we_epoch -1 ^
        --enable_update ^
        --update_start_epoch 350 ^
        --update_llm_model gpt-4o ^
        --dpo_topic_filter cv_below_avg ^
        --dpo_weight 20.0 ^
        --dpo_alpha 1.0 ^
        --contrastive_weight 20.0 ^
        --contrastive_ramp_epochs 0 ^
        --contrastive_topk 2 ^
        --contrastive_temperature 0.5 ^
        --contrastive_queue_size 0 ^
        --contrastive_doc_encoder BAAI/bge-small-en-v1.5 ^
        --contrastive_loss_type supcon ^
        --doc_embedding_source rebuild_from_text ^
        --enable_llm_eval

      if errorlevel 1 (
        echo [ERROR] dataset=%%D num_topics=%%T failed. Exiting.
        exit /b 1
      )
    )
  )
)

echo All runs finished.
endlocal
