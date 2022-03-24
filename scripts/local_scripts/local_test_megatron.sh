#!/bin/bash

TRAIN_DATA=glue_dataset/data/QQP/train.tsv
VALID_DATA=glue_dataset/data/QQP/test.tsv
VOCAB_FILE=glue_dataset/data/bert-large-cased-vocab.txt

python ../Megatron-LM/tasks/main.py \
       --task QQP \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 2 \
       --global-batch-size 16 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 50 \
       --train-data $TRAIN_DATA \
       --valid-data $VALID_DATA \
       --tokenizer-type BertWordPieceLowerCase\
       --vocab-file $VOCAB_FILE \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 1 \
       --eval-interval 1000 \
       --epochs 1 \
       --eval-iters 1