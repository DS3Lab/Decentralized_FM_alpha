cd ~/GPT-home-private
source activate pytorch_p38
ip=$1
world_size=$2
rank=$3
cuda_id=$4
timestamp=$(date +%Y_%m_%d_%H_%M)

#central_ps, sharded_ps, allreduce
dp_mode=central_ps

# Change the script here for different settings.
ga_step=8
num_layers=5
batch_size=62

let "global_batch_size = $ga_step*$batch_size*8"

DIST_CONF="--pp-mode gpipe --dp-mode $dp_mode --gradient-accumulate-step $ga_step --world-size $world_size --pipeline-group-size 8 --data-group-size 8 --rank $rank --cuda-id $cuda_id"
MODEL_CONF="--embedding-dim 2048 --num-heads 16 --num-layers $num_layers --batch-size $batch_size --micro-batch-size 1"

if [ "$world_size" -ne 64 ]
then
  echo "Not correct number of nodes"
  exit 1
fi

log_path="./logs/${timestamp}_gpt3_xl_pp8_dp8_l${num_layers}_b${global_batch_size}_rank${rank}"

if [ $# -eq 4 ]
then
  python dist_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF >> "${log_path}_default.log"
elif [ $# -eq 5 ]
then
  case=$5
  export NCCL_SOCKET_IFNAME=ens3
  export GLOO_SOCKET_IFNAME=ens3
  sh ./scripts/tc_scripts/heterogeneous_setup_case"$case".sh
  python dist_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF >> "${log_path}_heter${case}.log"
  sh ./scripts/tc_scripts/clear.sh
elif [ $# -eq 6 ]
then
  DELAY_MS=$5
  RATE_GBIT=$6
  export NCCL_SOCKET_IFNAME=ens3
  export GLOO_SOCKET_IFNAME=ens3
  sh ./scripts/tc_scripts/both_delay_bandwidth.sh $DELAY_MS $RATE_GBIT
  python dist_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF --trace-postfix "d${DELAY_MS}b${RATE_GBIT}" >> "${log_path}_d${DELAY_MS}b${RATE_GBIT}.log"
  sh ./scripts/tc_scripts/clear.sh
else
  echo "Invalid argument number!"
fi

echo "Benchmark training is done."