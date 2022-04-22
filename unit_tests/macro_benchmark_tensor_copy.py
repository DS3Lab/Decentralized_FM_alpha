import torch

size = 1024 * 1024 * 1
test_tensor_gpu = torch.ones(size, dtype=torch.float16, device='cuda:0')
test_tensor_cpu = torch.zeros(size, dtype=torch.float32, device='cpu')

cpu_to_gpu_stream = torch.cuda.Stream(device='cuda:0')
gpu_to_cpu_stream = torch.cuda.Stream(device='cuda:0')

copy_to_cpu_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
copy_to_cpu_ready_event = torch.cuda.Event(enable_timing=True, blocking=False)

with torch.cuda.stream(gpu_to_cpu_stream):
    gpu_to_cpu_stream.record_event(copy_to_cpu_start_event)
    test_tensor_cpu.copy_(test_tensor_gpu, non_blocking=True)
    gpu_to_cpu_stream.record_event(copy_to_cpu_ready_event)

print(test_tensor_gpu)
print(test_tensor_cpu)
print("Copy from GPU to CPU time:", copy_to_cpu_start_event.elapsed_time(copy_to_cpu_ready_event))

test_tensor_cpu = test_tensor_cpu * 2

copy_to_gpu_start_event = torch.cuda.Event(enable_timing=True, blocking=False)
copy_to_gpu_ready_event = torch.cuda.Event(enable_timing=True, blocking=False)

with torch.cuda.stream(cpu_to_gpu_stream):
    cpu_to_gpu_stream.record_event(copy_to_gpu_start_event)
    test_tensor_gpu.copy_(test_tensor_cpu, non_blocking=True)
    cpu_to_gpu_stream.record_event(copy_to_gpu_ready_event)
print(test_tensor_gpu)
print(test_tensor_cpu)
print("Copy from CPU to GPU time:", copy_to_gpu_start_event.elapsed_time(copy_to_gpu_ready_event))

layer = torch.nn.Linear(5, 5)

print(layer.weight)

with torch.cuda.stream(cpu_to_gpu_stream):
    layer.to(device='cuda:0', non_blocking=True)

print(layer.weight)