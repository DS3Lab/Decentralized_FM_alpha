source ./ip_list.sh

world_size=${#ips[@]}

script=$1


for rank in "${!ips[@]}"
do
  echo "Issue command $script$ in Rank-$rank node: ${ips[$rank]}"
  if [ $# -eq 1 ]
  then
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" &
  elif [ $# -eq 3 ]
  then
    delay_ms=$2
    rate_gbit=$3
    ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/"${script}" "$master_ip" "$world_size" "$rank" "$delay_ms" "$rate_gbit" &
  else
    echo "Error! Not valid arguments."
  fi
done