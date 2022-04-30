cd cd ~/GPT-home-private/scheduler/
rank=$1
world_size=$2
case=$3

python generate_heterogeneous_tc.py --case 5 --rank $rank --nodes $world_size