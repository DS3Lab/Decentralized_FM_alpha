source ./ip_list.sh

#nodes_per_node=(8 8 8 8 8 8 8 8)
nodes_per_node=(2 2 2 2 2 2 2 2)
world_size=64

script=$1

declare -i rank=0
for node_rank in "${!ips[@]}"
do
  echo "Issue command $script in Rank-${node_rank} node: ${ips[node_rank]}"
  for (( i=0; i<${nodes_per_node[$node_rank]}; i++))
  do
    sshpass -p password ssh fsuser@"${ips[node_rank]}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" "$i" &
    rank+=1
  done
done
wait