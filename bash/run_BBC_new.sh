#!/usr/bin/env bash

python main.py --dataset BBC_new --n_topic 25 --weight_loss_ECR 150
python main.py --dataset BBC_new --n_topic 50 --weight_loss_ECR 250
python main.py --dataset BBC_new --n_topic 75 --weight_loss_ECR 300
python main.py --dataset BBC_new --n_topic 100 --weight_loss_ECR 300
