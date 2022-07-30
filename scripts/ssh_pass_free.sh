source ./ip_list.sh

rm ./ds_pwfree_ssh/id_rsa*

for rank in "${!ips[@]}"
do
  echo "Issue command in "${ips[rank]}" to generate rsa_key"
  ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/local_ssh_key_gen.sh "$rank" &
done
wait

for rank in "${!ips[@]}"
do
  echo "Download key file"
  scp -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}":~/.ssh/id_rsa_r"${rank}".pub ./ds_pwfree_ssh/ &
done
wait

for rank in "${!ips[@]}"
do
  echo "Upload key file"
  scp -i ../binhang_ds3_aws_oregon.pem ./ds_pwfree_ssh/id_rsa_r*.pub ubuntu@"${ips[rank]}":~/.ssh/ &
done
wait

world_size=${#ips[@]}
for rank in "${!ips[@]}"
do
  echo "Issue command in "${ips[rank]}" to generate setup ssh"
  ssh -i ../binhang_ds3_aws_oregon.pem ubuntu@"${ips[rank]}" "bash -s" < ./local_scripts/local_ssh_key_setup.sh "$world_size" &
done
wait