cd ~/GPT-home-private
source activate pytorch_p38
ip=$1
world_size=$2
rank=$3

timestamp=$(date +%Y_%m_%d_%H_%M)

if [ "$world_size" -ne 12 ]
then
  exit 1
fi

if [ $# -eq 3 ]
then
  python dist_fairscale_zero_s3.py  --dist-url tcp://"$ip":9000 --dist-backend nccl --world-size "$world_size" --rank "$rank" --embedding-dim 768 --num-heads 12 --num-layers 12 --batch-size 2 >> "./logs/${timestamp}_gpt3_small_fsdp12_default.log"
elif [ $# -eq 5 ]
then
  DELAY_MS=$4
  RATE_GBIT=$5
  export NCCL_SOCKET_IFNAME=ens3
  export GLOO_SOCKET_IFNAME=ens3
  sh ./scripts/tc_scripts/both_delay_bandwidth.sh $DELAY_MS $RATE_GBIT
  python dist_fairscale_zero_s3.py  --dist-url tcp://"$ip":9000 --dist-backend nccl --world-size "$world_size"  --rank "$rank" --embedding-dim 768 --num-heads 12 --num-layers 12 --batch-size 2 >> "./logs/${timestamp}_gpt3_small_fsdp12_d${DELAY_MS}b${RATE_GBIT}.log"
  sh ./scripts/tc_scripts/clear.sh
else
  echo "Invalid argument number!"
fi

echo "Benchmark training is done."