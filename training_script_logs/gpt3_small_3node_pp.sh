python3 dist_runner.py --dist-url tcp://172.31.33.81:9000 --mode gpipe --world-size 3 --pipeline-group-size 3 --data-group-size 1 --embedding-dim 768 --num-heads 12 --num-layers 4 --rank 0 --batch-size 64 --micro-batch-size 4