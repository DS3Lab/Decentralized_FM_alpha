source ./ip_list.sh

nodes_per_node=(8 8 8 8 8 8 8 8)
world_size=64

script=$1

declare -i rank=0
for node_rank in "${!ips[@]}"
do
  echo "Issue command $script in Rank-$rank node: ${ips[node_rank]}"
  if [ $# -eq 1 ]
  then
    echo "Running in default network."
    for (( i; i<${nodes_per_node[$node_rank]}; i++))
    do
      ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" 0 &
      rank+=1
    done
  elif [ $# -eq 2 ]
  then
    case=$2
    echo "Running in heterogeneous network: Case-$case"
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" 0 "$case" &
  elif [ $# -eq 3 ]
  then
    delay_ms=$2
    rate_gbit=$3
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" 0 "$delay_ms" "$rate_gbit" &
  else
    echo "Error! Not valid arguments."
  fi
done