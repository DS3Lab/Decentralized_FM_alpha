source ./ip_list.sh

for ip in "${ips[@]}"
do
  echo "Issue command in $ip"
  ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"$ip" '
    cd ~/GPT-home-private
    git pull
  '
done