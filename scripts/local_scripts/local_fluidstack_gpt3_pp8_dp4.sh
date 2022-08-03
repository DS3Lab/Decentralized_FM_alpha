cd ~/GPT-home-private

export NCCL_SOCKET_IFNAME=enp1s0

ip=$1
world_size=$2
rank=$3
cuda_id=$4
timestamp=$(date +%Y_%m_%d_%H_%M)

#central_ps, sharded_ps, allreduce
dp_mode=central_ps

# Change the script here for different settings.
############################################################
ga_step=2
num_layers=3
batch_size=128
############################################################

let "global_batch_size = $ga_step*$batch_size*4"
echo "Batch size: $global_batch_size"

DIST_CONF="--rank $rank --cuda-id $cuda_id --pp-mode gpipe --dp-mode $dp_mode --gradient-accumulate-step $ga_step --world-size $world_size --pipeline-group-size 8 --data-group-size 4"
MODEL_CONF="--embedding-dim 2048 --num-heads 16 --num-layers $num_layers --batch-size $batch_size --micro-batch-size 1"



log_mode=$8
log_path="./logs/${timestamp}_gpt3_xl_pp8_dp8_l${num_layers}_b${global_batch_size}_rank${rank}_${log_mode}"

python3 dist_training_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF >> "./logs/${timestamp}_rank${rank}_GPTJ_default.log"