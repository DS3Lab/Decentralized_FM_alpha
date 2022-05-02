source ./ip_list.sh

case=$1

for ip in "${ips[@]}"
do
  echo "Issue command in $ip"
  ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"$ip" "bash -s" < ./tc_scripts/heterogeneous_setup_case"$case".sh &
done