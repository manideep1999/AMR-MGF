#!/bin/bash

# # * laptop (with spring)
#  python ./train.py --model_name EHFB --dataset laptop_spring --seed 42 --bert_lr 2e-5 --batch_size 32 --num_epoch 15 --bert_dim 768 --max_length 100 --gamma 1 --theta 0.5\
#                     --max_num_spans 6 --dep_layers 3 --sem_layers 6 --amr_layers 6 --fusion_condition HF 

# # * laptop (with amrbart)
#  python ./train.py --model_name EHFB --dataset laptop_amrbart --seed 42 --bert_lr 2e-5 --batch_size 32 --num_epoch 15 --bert_dim 768 --max_length 100 --gamma 1 --theta 0.5\
#                     --max_num_spans 6 --dep_layers 3 --sem_layers 6 --amr_layers 6 --fusion_condition HF 

# * restaurants (with spring)
#  python ./train.py --model_name EHFB --dataset restaurant_spring --seed 42 --bert_lr 2e-5 --batch_size 32 --num_epoch 20 --bert_dim 768 --max_length 100 \
#                     --max_num_spans 6 --dep_layers 6 --sem_layers 9 --amr_layers 9 --fusion_condition HF --gamma 1 --theta 0.5

# * restaurants (with amrbart)
 python ./train.py --model_name EHFB --dataset restaurant_amrbart --seed 42 --bert_lr 5e-5 --batch_size 32 --num_epoch 20 --bert_dim 768 --max_length 100 \
                    --max_num_spans 6 --dep_layers 6 --sem_layers 9 --amr_layers 9 --fusion_condition HF --gamma 1 --theta 0.5 

# * twitter (with spring)
# python ./train.py --model_name EHFB --dataset twitter_spring --seed 100 --bert_lr 2e-5 --batch_size 16 --num_epoch 20 --bert_dim 768 --max_length 100 \
#                   --max_num_spans 3 --dep_layers 3 --sem_layers 1 --amr_layers 1 --fusion_condition HF --gamma 1 --theta 0.07

# * twitter (with amrbart)
# python ./train.py --model_name EHFB --dataset twitter_amrbart --seed 1000 --bert_lr 2e-5 --batch_size 16 --num_epoch 20 --bert_dim 768 --max_length 100 \
#                   --max_num_spans 3 --dep_layers 3 --sem_layers 1 --amr_layers 1 --fusion_condition HF --gamma 1 --theta 0.07


