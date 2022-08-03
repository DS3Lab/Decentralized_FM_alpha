source ./ip_list.sh

world_size=${#ips[@]}

script=$1

PIPELINE_PARALLEL_SIZE=$2
TENSOR_PARALLEL_SIZE=$3
GPUS_PER_NODE=$4
num_layers=$5
global_batch_size=$6

for rank in "${!ips[@]}"
do
  ip=${ips[rank]}
  echo "Issue command $script in Rank-$rank node: ${ips[$rank]}"
  ssh fsuser@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}" 1 "$PIPELINE_PARALLEL_SIZE" "$TENSOR_PARALLEL_SIZE" "$master_ip" "$world_size" "$rank" "$num_layers" "$global_batch_size"&

done
wait