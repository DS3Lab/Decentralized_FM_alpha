cd ~/GPT-home-private
source activate pytorch_p38
ip=$1
world_size=$2
rank=$3
timestamp=$(date +%Y_%m_%d_%H_%M)

DIST_CONF="--pp-mode gpipe --world-size $world_size --pipeline-group-size $world_size --data-group-size 1 --rank "$rank""
INFERENCE_CONF="--batch-size 64 --input-seq-length 128 --generate-seq-length 128 --micro-batch-size 1 --num-layers 4"


if [ "$world_size" -ne 3 ]
then
  echo "Not correct number of nodes"
  exit 1
fi


python dist_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF>> "./logs/${timestamp}_inference_pp3_default.log"