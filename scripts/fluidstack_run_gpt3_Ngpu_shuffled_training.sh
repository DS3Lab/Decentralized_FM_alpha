source ./ip_list.sh

nodes_per_node=(2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2)
world_size=32

script=$1
num_layer=$2
ga_step=$3

declare -i rank_index=0

# Random seed 2024
rank_map=(0 27 16 6 12 9 8 7 25 21 1 28 18 22 3 14 20 26 17 11 2 23 24 15 13 29 31 10 19 5 30 4)

for node_rank in "${!ips[@]}"
do
  echo "Issue command $script in Rank-${node_rank} node: ${ips[node_rank]}"
    for (( i=0; i<${nodes_per_node[$node_rank]}; i++))
    do
      echo ${rank_map[rank_index]}
      ssh fsuser@"${ips[node_rank]}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "${rank_map[rank_index]}" "$i" "$num_layer" "$ga_step" "wo_scheduler" &
      rank_index+=1
    done
done

wait
