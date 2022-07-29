cd ~/GPT-home-private

source activate pytorch_p38
echo "Batch size 1:"
python local_175B_cpu_inference.py --skip-prompt --fp16 --num-layers 96  --batch-size 1 >> ./logs/local_175B_cpu_inference.log

echo "Batch size 2:"
python local_175B_cpu_inference.py --skip-prompt --fp16 --num-layers 96  --batch-size 2 >> ./logs/local_175B_cpu_inference.log

echo "Batch size 4:"
python local_175B_cpu_inference.py --skip-prompt --fp16 --num-layers 96  --batch-size 4 >> ./logs/local_175B_cpu_inference.log

echo "Batch size 8:"
python local_175B_cpu_inference.py --skip-prompt --fp16 --num-layers 96  --batch-size 8 >> ./logs/local_175B_cpu_inference.log

echo "Batch size 16:"
python local_175B_cpu_inference.py --skip-prompt --fp16 --num-layers 96  --batch-size 16 >> ./logs/local_175B_cpu_inference.log

echo "Batch size 32:"
python local_175B_cpu_inference.py --skip-prompt --fp16 --num-layers 96  --batch-size 32 >> ./logs/local_175B_cpu_inference.log

echo "Batch size 48:"
python local_175B_cpu_inference.py --skip-prompt --fp16 --num-layers 96  --batch-size 48 >> ./logs/local_175B_cpu_inference.log

echo "Batch size 64:"
python local_175B_cpu_inference.py --skip-prompt --fp16 --num-layers 96  --batch-size 64 >> ./logs/local_175B_cpu_inference.log