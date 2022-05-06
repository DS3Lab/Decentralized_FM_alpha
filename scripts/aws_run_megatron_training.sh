source ./ip_list.sh

world_size=${#ips[@]}

script=$1

PIPELINE_PARALLEL_SIZE=$2
TENSOR_PARALLEL_SIZE=$3
GPUS_PER_NODE=$4

num_layers=$5
global_batch_size=$6

rank_map=(0 2 32 33 4 10 7 45 36 8 51 26 11 5 53 1 40 23 37 14 13 43 54 21 57 35 63 18 6 24 16 22 38 3 58 61 44 27 52 30 15 9 39 47 48 41 31 20 12 28 34 42 17 55 19 25 56 60 59 50 49 46 29 62)


for i in "${!ips[@]}"
do
  rank=${rank_map[$i]}
  ip=${ips[i]}
  echo "Issue command $script in Rank-$rank node: ${ips[$rank]}"
  if [ $rank -eq 63 ]
  then
    echo "========Last rank IP ${ip}==========="
  fi
  if [ $# -eq 6 ]
  then
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}"  "$PIPELINE_PARALLEL_SIZE" "$TENSOR_PARALLEL_SIZE" "$master_ip" "$GPUS_PER_NODE" "$world_size" "$rank" "$num_layers" "$global_batch_size"&
  elif [ $# -eq 7 ]
  then
    case=$7
    echo "Running in heterogeneous network: Case-$case"
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}"  "$PIPELINE_PARALLEL_SIZE" "$TENSOR_PARALLEL_SIZE" "$master_ip" "$GPUS_PER_NODE" "$world_size" "$rank"  "$num_layers" "$global_batch_size" "$case" &
  elif [ $# -eq 8 ]
  then
    delay_ms=$7
    rate_gbit=$8
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}"  "$PIPELINE_PARALLEL_SIZE" "$TENSOR_PARALLEL_SIZE" "$master_ip" "$GPUS_PER_NODE" "$world_size" "$rank"  "$num_layers" "$global_batch_size" "$delay_ms" "$rate_gbit" &
  else
    echo "Error! Not valid arguments."
  fi
done
wait