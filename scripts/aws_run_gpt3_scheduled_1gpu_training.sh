source ./ip_list.sh

world_size=${#ips[@]}

script=$1

# This is need to training scheduled examples. Use generate_assignment.py to generate them (copy the result printout here).
rank_map=(0 2 32 33 4 10 7 45 36 8 51 26 11 5 53 1 40 23 37 14 13 43 54 21 57 35 63 18 6 24 16 22 38 3 58 61 44 27 52 30 15 9 39 47 48 41 31 20 12 28 34 42 17 55 19 25 56 60 59 50 49 46 29 62)


ga_step=$2
num_layers=$3
batch_size=$4

log_mode='random_map'

for i in "${!ips[@]}"
do
  rank=${rank_map[$i]}
  ip=${ips[i]}
  echo "Issue command $script in Rank-$rank node: ${ips[$rank]}"
  if [ $# -eq 4 ]
  then
    echo "Running in default network."
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"$ip" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" 0 "$ga_step" "$num_layers" "$batch_size" "$log_mode" &
  elif [ $# -eq 5 ]
  then
    case=$5
    echo "Running in heterogeneous network: Case-$case"
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"$ip" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" 0 "$ga_step" "$num_layers" "$batch_size" "$log_mode" "$case" &
  elif [ $# -eq 6 ]
  then
    delay_ms=$5
    rate_gbit=$6
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"$ip" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" 0 "$ga_step" "$num_layers" "$batch_size" "$log_mode" "$delay_ms" "$rate_gbit" &
  else
    echo "Error! Not valid arguments."
  fi
done
wait