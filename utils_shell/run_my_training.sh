source ./ip_list.sh

for rank in "${!ips[@]}"
do
  echo "Issue command in Rank-$rank node: ${ips[$rank]}"
  ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_start_python_script.sh "$rank" &
done