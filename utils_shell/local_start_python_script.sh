cd ~/GPT-home-private
    source activate pytorch_p38
    python dist_runner.py --dist-url tcp://172.31.41.25:9000 --mode gpipe --world-size 2 --pipeline-group-size 2 --data-group-size 1 --embedding-dim 768 --num-heads 12 --num-layers 2 --rank "$1" --batch-size 32 --micro-batch-size 4 > temp_log.txt