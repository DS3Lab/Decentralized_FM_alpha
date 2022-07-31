hostnames=$1

for hostname in "${hostnames[@]}"
do
  echo "Add "$hostname" in ssh known hosts"
  ssh-keyscan -H "$hostname" >> ~/.ssh/known_hosts
done