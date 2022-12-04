
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

python -u slurms_scrips/eval_opt.py --ckpt-root {{ckpt_folder}} --output-path {{output_path}} --step-set {{step_set}}
'''

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", default='opt-allreduce', help="ckpts")
    args = parser.parse_args()
    
    step_set = [
        "checkpoint_100",
        "checkpoint_200",
        "checkpoint_300",
        "checkpoint_500",
        "checkpoint_1000",
        "checkpoint_1500",
        "checkpoint_2000",
        "checkpoint_2500",
        "checkpoint_3000",
        "checkpoint_3500",
        "checkpoint_4000",
        "checkpoint_4500",
        "checkpoint_5000",
    ]
    
    template = template.replace("{{ckpt_folder}}", f"/nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm/pretrained_models/checkpoints/{args.ckpt_root}")
    template = template.replace("{{output_path}}", f"result-{args.ckpt_root}.jsonl")
    template = template.replace("{{step_set}}", ';'.join(step_set))
    
    with open('slurms_scrips/eval_to_submit.lsf.sh', 'w') as f:
        f.write(template)

    os.system('bsub < slurms_scrips/eval_to_submit.lsf.sh')
    