cd ~/GPT-home-private
source activate pytorch_p38
ip=$1
world_size=$2
rank=$3

timestamp=$(date +%Y_%m_%d_%H_%M)

#central_ps, sharded_ps, allreduce
dp_mode=central_ps
DIST_CONF="--pp-mode gpipe --dp-mode $dp_mode --gradient-accumulate-step 2 --world-size $world_size --pipeline-group-size 8 --data-group-size 8 --rank "$rank""
MODEL_CONF="--embedding-dim 2048 --num-heads 16 --num-layers 5 --batch-size 63 --micro-batch-size 1"

if [ "$world_size" -ne 64 ]
then
  echo "Not correct number of nodes"
  exit 1
fi

if [ $# -eq 3 ]
then
  python dist_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF >> "./logs/${timestamp}_gpt3_xxl_pp8_dp8_default.log"
elif [ $# -eq 5 ]
then
  DELAY_MS=$4
  RATE_GBIT=$5
  export NCCL_SOCKET_IFNAME=ens3
  export GLOO_SOCKET_IFNAME=ens3
  sh ./scripts/tc_scripts/both_delay_bandwidth.sh $DELAY_MS $RATE_GBIT
  python dist_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF --trace-postfix "d${DELAY_MS}b${RATE_GBIT}" >> "./logs/${timestamp}_gpt3_xxl_pp8_dp8_d${DELAY_MS}b${RATE_GBIT}.log"
  sh ./scripts/tc_scripts/clear.sh
else
  echo "Invalid argument number!"
fi

echo "Benchmark training is done."