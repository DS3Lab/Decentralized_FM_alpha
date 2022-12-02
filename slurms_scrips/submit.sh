
job_id=`python3 -c 'import uuid; print(uuid.uuid4())'`
pp_degree=4
dp_degree=1
n_layer_per_device=1

world_size=`expr $pp_degree \* $dp_degree`

for((i=0;i<${world_size};++i))
do

    echo submitting ${i}-th job
    sbatch slurms_scrips/train_template.slurm.sh ${job_id} ${pp_degree} ${dp_degree} ${n_layer_per_device}

done