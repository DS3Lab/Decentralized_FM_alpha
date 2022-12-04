
import os
import uuid

template = '''#!/bin/bash
#BSUB -n 4                     # 1 cores
#BSUB -W 23:59                   # 3-minutes run-time
#BSUB -R "rusage[mem=8000]"     # 32 GB per core
#BSUB -R "rusage[ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=20000]"
#BSUB -o /cluster/home/juewang/fm/juewang/exe_log/out.%J
#BSUB -e /cluster/home/juewang/fm/juewang/exe_log/err.%J

module load gcc/6.3.0 cuda/11.0.3 eth_proxy       # Load modules from Euler setup
source activate pipeline                          # Activate my conda python environment
cd /cluster/home/juewang/fm/juewang/Decentralized_FM_alpha_train     # Change directory

nvidia-smi

ckpt_folder=/nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm/pretrained_models/checkpoints/opt-allreduce

python -u slurms_scrips/eval_opt.py --ckpt-root ${ckpt_folder}
'''

if __name__ == '__main__':
    
    with open('slurms_scrips/eval_to_submit.lsf.sh', 'w') as f:
        f.write(template)

    os.system('bsub < slurms_scrips/eval_to_submit.lsf.sh')
    