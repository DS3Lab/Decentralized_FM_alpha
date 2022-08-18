# Run GPT@Home on Euler HPC cluster in ETH Zurich


- Check submitted job states:

      bjobs

Go to this directory in the Euler cluster:

- Load Lib test:

      bsub < lsf_foo.bsub

- Local training test:

      bsub < lsf_local_train_test.bsub


- The current repo in the Euler cluster:

      /nfs/iiscratch-zhang.inf.ethz.ch/export/zhang/export/fm/GPT-home-private 



- To check what is going on in a node,

       bjob_connect job_ID