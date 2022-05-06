source ./ip_list.sh

world_size=${#ips[@]}

script=$1

MICRO_BATCH_SIZE=$2
PIPELINE_PARALLEL_SIZE=$3
TENSOR_PARALLEL_SIZE=$4
GPUS_PER_NODE=$5



for rank in "${!ips[@]}"
do
  echo "Issue command $script in Rank-$rank node: ${ips[$rank]}"
  if [ $# -eq 5 ]
  then
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}"  "$MICRO_BATCH_SIZE" "$PIPELINE_PARALLEL_SIZE" "$TENSOR_PARALLEL_SIZE" "$master_ip" "$GPUS_PER_NODE" "$world_size" "$rank" &
  elif [ $# -eq 6 ]
    case=$6
    echo "Running in heterogeneous network: Case-$case"
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}"  "$MICRO_BATCH_SIZE" "$PIPELINE_PARALLEL_SIZE" "$TENSOR_PARALLEL_SIZE" "$master_ip" "$GPUS_PER_NODE" "$world_size" "$rank" &

  elif [ $# -eq 7 ]
  then
    delay_ms=$6
    rate_gbit=$7
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}" "$MICRO_BATCH_SIZE" "$PIPELINE_PARALLEL_SIZE" "$TENSOR_PARALLEL_SIZE" "$master_ip" "$GPUS_PER_NODE" "$world_size" "$rank" "$delay_ms" "$rate_gbit" &
  else
    echo "Error! Not valid arguments."
  fi
done
wait