GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
MICRO_BATCH_SIZE=$1

FIXED_MODEL_CONFIG="--embedding-dim 2048 --num-heads 16 --num-layers 2 --batch-size 64"
DIST_CONFIG="--dist-url tcp://$MASTER_ADDR:$MASTER_PORT --mode gpipe --world-size $GPUS_PER_NODE --pipeline-group-size $GPUS_PER_NODE --data-group-size 1"

for rank in `seq 0 $GPUS_PER_NODE-1`
do
  echo "Issue cmd on GPU-$rank"
  if [ $rank -eq 0 ]
  then
    python dist_runner.py  $FIXED_MODEL_CONFIG $DIST_CONFIG --rank "$rank" --cuda-id "$rank" --micro-batch-size $MICRO_BATCH_SIZE
  else
    python dist_runner.py  $FIXED_MODEL_CONFIG $DIST_CONFIG --rank "$rank" --cuda-id "$rank" --micro-batch-size $MICRO_BATCH_SIZE >> "./logs/foo_$rank.log"
  fi &
done