case=$1
batches=(1024 2048 4096)
layers=(24 32 40)

for batch in "${batches[@]}"
do
  for layer in "${layers[@]}"
  do
    echo "$batch, $layer"
    bash aws_run_megatron_training.sh megatron_gpt3_xl_mp8_dp8.sh 8 1 1 "$layer" "$batch" "$case"
    sleep 10s
  done
done

bash copy_rank_last_logs.sh