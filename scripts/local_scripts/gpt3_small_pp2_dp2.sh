cd ~/GPT-home-private
source activate pytorch_p38
ip=$1
world_size=$2
rank=$3

#central_ps, sharded_ps, allreduce
dp_mode=sharded_ps

timestamp=$(date +%Y_%m_%d_%H_%M)

if [ "$world_size" -ne 4 ]
then
  exit 1
fi

if [ $# -eq 3 ]
then
  python dist_runner.py --dist-url tcp://"$ip":9000 --fp16 --pp-mode gpipe --dp-mode "$dp_mode" --world-size "$world_size" --pipeline-group-size 2 --data-group-size 2 --rank "$rank" --embedding-dim 768 --num-heads 12 --num-layers 6 --batch-size 64 --micro-batch-size 1 >> "./logs/${timestamp}_gpt3_small_gpipe_${dp_mode}_pp2_dp2_default.log"
elif [ $# -eq 5 ]
then
  DELAY_MS=$4
  RATE_GBIT=$5
  export NCCL_SOCKET_IFNAME=ens3
  export GLOO_SOCKET_IFNAME=ens3
  sh ./scripts/tc_scripts/both_delay_bandwidth.sh $DELAY_MS $RATE_GBIT
  python dist_runner.py --dist-url tcp://"$ip":9000 --fp16 --pp-mode gpipe --dp-mode "$dp_mode" --world-size "$world_size" --pipeline-group-size 2 --data-group-size 2 --rank "$rank" --embedding-dim 768 --num-heads 12 --num-layers 6 --batch-size 64 --micro-batch-size 1 --trace-postfix "d${DELAY_MS}b${RATE_GBIT}" >> "./logs/${timestamp}_gpt3_small_gpipe_${dp_mode}_pp2_dp2_d${DELAY_MS}b${RATE_GBIT}.log"
  sh ./scripts/tc_scripts/clear.sh
else
  echo "Invalid argument number!"
fi

echo "Benchmark training is done."