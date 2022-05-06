case=$1
batches=(1024 2048 4096)
layers=(24 32 40)

for layer in "${layers[@]}"
do
  for batch in "${batches[@]}"
  do
    echo "$batch, $layer"
    bash aws_run_megatron_training.sh megatron_gpt3_xl_mp8_dp8.sh 8 1 1 "$layer" "$batch" "$case"
    sleep 10s
  done
done

# bash aws_run_megatron_training.sh megatron_gpt3_xl_mp8_dp8.sh 1 8 1 24 1024 "$case"
# sleep 10s

# bash aws_run_megatron_training.sh megatron_gpt3_xl_mp8_dp8.sh 4 2 1 24 1024 "$case"
# sleep 10s

# bash aws_run_megatron_training.sh megatron_gpt3_xl_mp8_dp8.sh 2 4 1 24 1024 "$case"
# sleep 10s

bash copy_rank_last_logs.sh