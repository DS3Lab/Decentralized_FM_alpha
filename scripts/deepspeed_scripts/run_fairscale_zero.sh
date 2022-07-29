# this is the same as standard PyTorch:
python dist_fairscale_zero_s3.py --dist-url tcp://172.31.38.62:9000 --dist-backend nccl --world-size 2 --rank 1