source ./ip_list.sh

world_size=${#ips[@]}

script=$1

# This is need to training scheduled examples. Use generate_random_assignment.py to generate them (copy the result printout here).
rank_map=(

)

for i in "${!ips[@]}"
do
  rank=${rank_map[$i]}
  ip=${ips[rank]}
  echo "Issue command $script in Rank-$rank node: ${ips[$rank]}"
  if [ $# -eq 1 ]
  then
    echo "Running in default network."
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"$ip" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" 0 &
  elif [ $# -eq 2 ]
  then
    case=$2
    echo "Running in heterogeneous network: Case-$case"
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"$ip" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" 0 "$case" &
  elif [ $# -eq 3 ]
  then
    delay_ms=$2
    rate_gbit=$3
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"$ip" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" 0 "$delay_ms" "$rate_gbit" &
  else
    echo "Error! Not valid arguments."
  fi
done