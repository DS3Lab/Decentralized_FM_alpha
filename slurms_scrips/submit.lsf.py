
import os
import uuid

if __name__ == '__main__':

    with open('slurms_scrips/train_template.lsf.sh') as f:
        template = f.read()

    job_id = uuid.uuid4()
    pp_degree=4
    dp_degree=1
    n_layer_per_device=1
    world_size = pp_degree * dp_degree

    template = template.replace('{{JOB_ID}}', job_id)
    template = template.replace('{{PP_DEGREE}}', pp_degree)
    template = template.replace('{{DP_DEGREE}}', dp_degree)
    template = template.replace('{{N_LAYER_PER_DEVICE}}', n_layer_per_device)

    with open('slurms_scrips/train_to_submmit.lsf.sh') as f:
        f.write(template)
        
    for i in range(world_size):
        os.system('bsub < slurms_scrips/train_to_submmit.lsf.sh')
    