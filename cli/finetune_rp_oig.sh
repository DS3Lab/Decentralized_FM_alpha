job_id=0

netif=lo
ip="127.0.0.1"

export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export NCCL_DEBUG=INFO
export PYTHONPATH=.
export SHOW_DATA=1
export WANDB_NAME=rp-oig-sn

main_program=dist_lm_pretrain.py

DATASETS="/root/ft_data/oig_clean.jsonl:1.0"

ARGS="--model-name /root/fm/models/redpajama/checkpoints_hf_rp_700b_real_fp16/ \
--tokenizer-name /root/fm/models/redpajama/checkpoints_hf_rp_700b_real_fp16/ \
--project-name rp-oig \
--optimizer adam \
--model-type gptneox \
--seed 42 \
--checkpoint-path ./model_checkpoints/$WANDB_NAME \
--load-pretrained-model true \
--task-name "${DATASETS}" \
--num-layers 8 --embedding-dim 4096 \
--total-steps 18500 --warmup-steps 500 --train-warmup-steps 0 \
--checkpoint-steps 500 \
--lr 1e-6 --seq-length 2048 --batch-size 64 --micro-batch-size 1 --gradient-accumulate-step 4 \
--dist-url tcp://${ip}:9017 \
--world-size 8 --pipeline-group-size 4 --data-group-size 2 \
--job-id ${job_id} --net-interface ${netif} \
--fp16 \
--dp-backend nccl \
--dp-mode allreduce \
--pp-mode gpipe --profiling no-profiling"

(trap 'kill 0' SIGINT; \
python ${main_program} $(echo ${ARGS}) --cuda-id 0 --rank 0 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 1 --rank 1 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 2 --rank 2 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 3 --rank 3 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 4 --rank 4 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 5 --rank 5 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 6 --rank 6 \
    & \
python ${main_program} $(echo ${ARGS}) --cuda-id 7 --rank 7 \
    & \
wait)
