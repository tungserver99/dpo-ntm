#!/usr/bin/env bash

python main.py --dataset WOS_vocab_10k --n_topic 25 --weight_loss_ECR 100
python main.py --dataset WOS_vocab_10k --n_topic 50 --weight_loss_ECR 100
python main.py --dataset WOS_vocab_10k --n_topic 75 --weight_loss_ECR 100
python main.py --dataset WOS_vocab_10k --n_topic 100 --weight_loss_ECR 100
