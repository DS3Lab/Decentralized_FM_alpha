cd ~/GPT-home-private
source activate pytorch_p38
ip=$1
world_size=$2
pipeline_size=$3
rank=$4
node_type=$5

stage_num_layers=2
global_num_layers=$((stage_num_layers*$pipeline_size))

timestamp=$(date +%Y_%m_%d_%H_%M)


DIST_CONF="--world-size $world_size --pipeline-group-size $pipeline_size --rank $rank --node-type $node_type"
MODEL_CONF="--model-type gptj --model-name ./pretrained_models/gpt-j-6B"
INFERENCE_CONF="--input-seq-length 512 --generate-seq-length 32 --prompt-micro-batch-size 1 --token-micro-batch-size 1 --stage-num-layers $stage_num_layers --global-num-layers $global_num_layers"
BUF_CONF="--producer-buffer-size 4 --consumer-buffer-size 4"

if [ "$world_size" -ne 6 ]
then
  echo "Not correct number of nodes"
  exit 1
fi


python dist_hybrid_inference_runner.py --dist-url tcp://"$ip":9000 --fp16 $DIST_CONF $MODEL_CONF $INFERENCE_CONF $BUF_CONF >> "./logs/${timestamp}_GPTJ_hybrid_inference_pp14_default.log"