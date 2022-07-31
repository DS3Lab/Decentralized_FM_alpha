cd ~/GPT-home-private
source activate pytorch_p38

batches=(1024,2048,4096)
layers=(24,32,40)

for batch in "${batches[@]}"
do
  for layer in "${layers[@]}"
  do
    deepspeed --hostfile=./scripts/ds_hostnames dist_deepspeed_zero_s3.py --embedding-dim 2048 --seq-length 2048 --batch-size $batch --num-layers $layer
  done
done

