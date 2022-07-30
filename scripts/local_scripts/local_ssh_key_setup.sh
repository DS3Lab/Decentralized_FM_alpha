world_size=$1

for (( i=0; i<"$world_size"; i++))
do
  cat ~/.ssh/id_rsa_r"$i".pub >> ~/.ssh/authorized_keys
done

sudo chmod -R 700 .ssh
sudo chmod -R 640 .ssh/authorized_keys
