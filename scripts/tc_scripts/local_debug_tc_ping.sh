# Update this list before use.
private_ips=(
"172.31.16.228"
"172.31.26.150"
"172.31.22.73"
"172.31.27.129"
"172.31.20.130"
"172.31.22.50"
"172.31.22.4"
"172.31.23.80"
)


for ip in "${private_ips[@]}"
do
  ping -c 5 ip
done
