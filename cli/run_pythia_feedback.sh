

job_id=0

netif=lo
ip="127.0.0.1"

export GLOO_SOCKET_IFNAME=${netif}
export NCCL_SOCKET_IFNAME=${netif}
export NCCL_DEBUG=INFO
export SHOW_DATA=1
export WANDB_NAME=pythia-6.9b-ock-feedback-5.0
export WANDB_ENTITY=asdfffjj

main_program=dist_lm_pretrain.py

ARGS="--model-name /root/fm/models/pythia-t-chat-v0.16 \
--tokenizer-name /root/fm/models/pythia-t-chat-v0.16 \
--project-name slot-sgd \
--optimizer adam \
--model-type gptneox \
--seed 42 \
--checkpoint-path ./model_checkpoints/$WANDB_NAME \
--load-pretrained-model true \
--task-name \
/root/ft_data/mix_v12.jsonl:1.0,\
ni_chat:0.1,\
/root/ft_data/unified_ni.jsonl:0.1,\
/root/ft_data/unified_p3.jsonl:0.1,\
/root/ft_data/unified_flan.jsonl:0.1,\
/root/ft_data/unified_chip2.jsonl:0.01,\
/root/ft_data/unified_oa_v3_fixed_plus_safety_fixed.jsonl:0.1,\
/root/ft_data/unified_soda_dialog.jsonl:0.1,\
/root/ft_data/unified_unifiedskg_instructions_v2.jsonl:0.1,\
/root/ft_data/unified_merged_code_xp3.jsonl:0.1,\
/root/ft_data/unified_oscar_en_sample_dialog.jsonl:0.1,\
/root/ft_data/unified_ul2_plus_oscar_en_sample_dialog.jsonl:0.1,\
/root/ft_data/unified_multi_news.jsonl:0.01,\
/root/ft_data/unified_openai_summarize_tldr.jsonl:0.01,\
/root/ft_data/unified_scitldr.jsonl:0.01,\
/root/ft_data/unified_squad_v2.jsonl:0.01,\
/root/ft_data/unified_nq.jsonl:0.01,\
/root/ft_data/unified_poetry_instructions.jsonl:0.01,\
/root/ft_data/unified_unatural_instructions.jsonl:0.01,\
/root/ft_data/unified_conv_finqa.jsonl:0.01,\
/root/ft_data/unified_essays.jsonl:0.01,\
/root/ft_data/unified_plot_screenplay_books_dialog.jsonl:0.01,\
/root/ft_data/unified_grade_school_math_instructions.jsonl:0.01,\
/root/ft_data/unified_cot_instructions.jsonl:0.01,\
/root/ft_data/unified_joke_explanations.jsonl:0.01,\
/root/ft_data/unified_cuad.jsonl:0.01,\
/root/ft_data/unified_abstact_infill.jsonl:0.1,\
/root/ft_data/unified_image_prompts_instructions.jsonl:0.01,\
/root/ft_data/unified_feedback_v0.2.jsonl:5.0 \
--num-layers 8 --embedding-dim 4096 \
--total-steps 400 --warmup-steps 10 --train-warmup-steps 0 \
--checkpoint-steps 100 \
--lr 1e-5 --seq-length 2048 --batch-size 16 --micro-batch-size 1 --gradient-accumulate-step 4 \
--dist-url tcp://${ip}:10187 \
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
