cd ~/GPT-home-private
source activate pytorch_p38
ip=$1
world_size=$2
rank=$3

timestamp=$(date +%Y_%m_%d_%H_%M)

#central_ps, sharded_ps, allreduce
dp_mode=central_ps

if [ "$world_size" -ne 16 ]
then
  echo "Not correct number of nodes"
  exit 1
fi

if [ $# -eq 3 ]
then
  python dist_runner.py --dist-url tcp://"$ip":9000 --fp16 --pp-mode gpipe --dp-mode "$dp_mode" --world-size "$world_size" --pipeline-group-size 8 --data-group-size 2 --rank "$rank" --embedding-dim 2048 --num-heads 16 --num-layers 3 --batch-size 96 --micro-batch-size 1 >> "./logs/${timestamp}_gpt3_xl_pp8_dp2_default.log"
elif [ $# -eq 5 ]
then
  DELAY_MS=$4
  RATE_GBIT=$5
  export NCCL_SOCKET_IFNAME=ens3
  export GLOO_SOCKET_IFNAME=ens3
  sh ./scripts/tc_scripts/both_delay_bandwidth.sh $DELAY_MS $RATE_GBIT
  python dist_runner.py --dist-url tcp://"$ip":9000 --fp16 --pp-mode gpipe --dp-mode "$dp_mode" --world-size "$world_size" --pipeline-group-size 8 --data-group-size 2 --rank "$rank" --embedding-dim 2048 --num-heads 16 --num-layers 3 --batch-size 96 --micro-batch-size 1 --trace-postfix "d${DELAY_MS}b${RATE_GBIT}" >> "./logs/${timestamp}_gpt3_xl_pp8_dp2_d${DELAY_MS}b${RATE_GBIT}.log"
  sh ./scripts/tc_scripts/clear.sh
else
  echo "Invalid argument number!"
fi

echo "Benchmark training is done."