source ./ip_list.sh


world_size=64

script=$1

# This is need to training scheduled examples. Use generate_random_assignment.py to generate them (copy the result printout here).
nodes_per_node=(8 8 8 8 8 8 8 8)
rank_map=(
(0 2 32 33 4 10 7 45)
(36 8 51 26 11 5 53 1)
(40 23 37 14 13 43 54 21)
(57 35 63 18 6 24 16 22)
(38 3 58 61 44 27 52 30)
(15 9 39 47 48 41 31 20)
(12 28 34 42 17 55 19 25)
(56 60 59 50 49 46 29 62)
)

for node_rank in "${!ips[@]}"
do
  echo "Issue command $script in Rank-$rank node: ${ips[node_rank]}"
  if [ $# -eq 1 ]
  then
    echo "Running in default network."
    for (( i=0; i<${nodes_per_node[$node_rank]}; i++))
    do
      ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[node_rank]}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "${rank_map[node_rank][i]}" "$i" &
    done
  elif [ $# -eq 2 ]
  then
    case=$2
    echo "Running in heterogeneous network: Case-$case"
    for (( i=0; i<${nodes_per_node[$node_rank]}; i++))
    do
      ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[node_rank]}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "${rank_map[node_rank][i]}" "$i" "$case" &
    done
  elif [ $# -eq 3 ]
  then
    delay_ms=$2
    rate_gbit=$3
    echo "Running homogeneous TC setting."
    for (( i=0; i<${nodes_per_node[$node_rank]}; i++))
    do
      ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[node_rank]}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "${rank_map[node_rank][i]}" "$i" "$delay_ms" "$rate_gbit" &
    done
  else
    echo "Error! Not valid arguments."
  fi
done