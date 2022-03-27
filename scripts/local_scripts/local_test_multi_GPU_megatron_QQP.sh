#!/bin/bash

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

VOCAB_FILE=glue_dataset/data/bert-large-cased-vocab.txt
TRAIN_FILE=glue_dataset/data/QQP/train.tsv
VALID_FILE=glue_dataset/data/QQP/dev.tsv
TEST_FILE=glue_dataset/data/QQP/test.tsv

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       ./dist_megatron_train_qqp.py \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 2 \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 16 \
       --micro-batch-size 2 \
       --global-batch-size 16 \
       --seq-length 512 \
       --max-position-embeddings 2048 \
       --tokenizer-type BertWordPieceLowerCase\
       --vocab-file $VOCAB_FILE \
       --train-data-path $TRAIN_FILE \
       --valid-data-path $VALID_FILE \
       --test-data-path $TEST_FILE \
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
       --eval-iters 1 \
       --no-bias-dropout-fusion