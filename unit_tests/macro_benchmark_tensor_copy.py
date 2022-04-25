import torch
import json
import cupy
import numpy


def copy_tensor_torch(n:int, dim:int, test_upload:bool, test_offload:bool):
    tensors_cpu = [torch.mul(torch.eye(dim, dtype=torch.float32, device='cpu'), 1 + 0.0005 * float(i)) for i in range(n)]
    for tensor in tensors_cpu:
        tensor = tensor.pin_memory()
    input_tensors_gpu = [torch.eye(dim, dtype=torch.float16, device='cuda:0') * i for i in range(n)]
    output_tensors_gpu = [torch.zeros((dim, dim), dtype=torch.float16, device='cuda:0') * i for i in range(n)]

    cpu_to_gpu_stream = torch.cuda.Stream(device='cuda:0', priority=-1)
    gpu_to_cpu_stream = torch.cuda.Stream(device='cuda:0', priority=-1)
    comp_stream = torch.cuda.default_stream(device='cuda:0')

    copy_to_gpu_start_events = [torch.cuda.Event(enable_timing=True, blocking=False) for _ in range(n)]
    copy_to_gpu_ready_events = [torch.cuda.Event(enable_timing=True, blocking=False) for _ in range(n)]
    gpu_compute_start_events = [torch.cuda.Event(enable_timing=True, blocking=False) for _ in range(n)]
    gpu_compute_ready_events = [torch.cuda.Event(enable_timing=True, blocking=False) for _ in range(n)]
    copy_to_cpu_start_events = [torch.cuda.Event(enable_timing=True, blocking=False) for _ in range(n)]
    copy_to_cpu_ready_events = [torch.cuda.Event(enable_timing=True, blocking=False) for _ in range(n)]

    start_event = torch.cuda.Event(enable_timing=True, blocking=False)
    end_event = torch.cuda.Event(enable_timing=True, blocking=False)

    for _ in range(3):
        torch.cuda.synchronize()
        start_event.record()
        for i in range(n):
            if test_upload:
                with torch.cuda.stream(cpu_to_gpu_stream):
                    cpu_to_gpu_stream.record_event(copy_to_gpu_start_events[i])
                    input_tensors_gpu[i].copy_(tensors_cpu[i], non_blocking=True)
                    cpu_to_gpu_stream.record_event(copy_to_gpu_ready_events[i])
            with torch.cuda.stream(comp_stream):
                comp_stream.wait_event(copy_to_gpu_ready_events[i])
                comp_stream.record_event(gpu_compute_start_events[i])
                output_tensors_gpu[i] = torch.matrix_power(input_tensors_gpu[i], 20)
                comp_stream.record_event(gpu_compute_ready_events[i])
            if test_offload:
                with torch.cuda.stream(gpu_to_cpu_stream):
                    gpu_to_cpu_stream.wait_event(gpu_compute_ready_events[i])
                    gpu_to_cpu_stream.record_event(copy_to_cpu_start_events[i])
                    # tensors_cpu[i].copy_(output_tensors_gpu[i], non_blocking=True)
                    tensors_cpu[i] = output_tensors_gpu[i].cpu()
                    gpu_to_cpu_stream.record_event(copy_to_cpu_ready_events[i])
        end_event.record()
        torch.cuda.synchronize()

    print("Total time: ", start_event.elapsed_time(end_event))

    for i in range(n):
        print(input_tensors_gpu[i])
        print(tensors_cpu[i])

    profile_logs = []

    for i in range(n):
        if test_upload:
            copy_to_gpu_ts = start_event.elapsed_time(copy_to_gpu_start_events[i]) * 1e+3
            copy_to_gpu_slot = copy_to_gpu_start_events[i].elapsed_time(copy_to_gpu_ready_events[i]) * 1e+3
            copy_to_gpu_log = {"name": "to_gpu", "ph": "X", "pid": 0, "tid": "1. Copy to GPU", "ts": copy_to_gpu_ts,
                               "dur": copy_to_gpu_slot, "args": {"micro-batch": i}, "cname": "startup"}
            profile_logs.append(copy_to_gpu_log)

        gpu_compute_ts = start_event.elapsed_time(gpu_compute_start_events[i]) * 1e+3
        gpu_compute_slot = gpu_compute_start_events[i].elapsed_time(gpu_compute_ready_events[i]) * 1e+3
        gpu_compute_log = {"name": "matrix_power", "ph": "X", "pid": 0, "tid": "2. GPU Compute", "ts": gpu_compute_ts,
                           "dur": gpu_compute_slot, "args": {"micro-batch": i}, "cname": "good"}
        profile_logs.append(gpu_compute_log)
        if test_offload:
            copy_to_cpu_ts = start_event.elapsed_time(copy_to_cpu_start_events[i]) * 1e+3
            copy_to_cpu_slot = copy_to_cpu_start_events[i].elapsed_time(copy_to_cpu_ready_events[i]) * 1e+3
            copy_to_cpu_log = {"name": "to_cpu", "ph": "X", "pid": 0, "tid": "3. Copy to CPU", "ts": copy_to_cpu_ts,
                               "dur": copy_to_cpu_slot, "args": {"micro-batch": i}, "cname": "thread_state_iowait"}
            profile_logs.append(copy_to_cpu_log)

    if test_upload and test_offload:
        path = '../trace_json/local_debug_CPU_GPU_torch_copy_both.json'
    elif test_upload:
        path = '../trace_json/local_debug_CPU_GPU_torch_copy_upload.json'
    elif test_offload:
        path = '../trace_json/local_debug_CPU_GPU_torch_copy_offload.json'
    else:
        assert False
    with open(path, 'w') as outfile:
        json.dump(profile_logs, outfile)


# This is from the original cupy lib.
def demo_cupy():
    def _pin_memory(array):
        mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
        ret = numpy.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
        ret[...] = array
        return ret
    SIZE = 1024 * 1024
    x_cpu_src = numpy.arange(SIZE, dtype=numpy.float32)
    x_gpu_src = cupy.arange(SIZE, dtype=numpy.float32)

    # synchronous
    stream = cupy.cuda.Stream.null
    start = stream.record()
    x_gpu_dst = cupy.empty(x_cpu_src.shape, x_cpu_src.dtype)
    x_gpu_dst.set(x_cpu_src)
    x_cpu_dst = x_gpu_src.get()
    end = stream.record()

    print('Synchronous Device to Host / Host to Device (ms)')
    print(cupy.cuda.get_elapsed_time(start, end))

    # asynchronous
    x_gpu_dst = cupy.empty(x_cpu_src.shape, x_cpu_src.dtype)
    x_cpu_dst = numpy.empty(x_gpu_src.shape, x_gpu_src.dtype)

    # x_pinned_cpu_src = _pin_memory(x_cpu_src)
    # x_pinned_cpu_dst = _pin_memory(x_cpu_dst)

    with cupy.cuda.stream.Stream() as stream_htod:
        start = stream_htod.record()
        # x_gpu_dst.set(x_pinned_cpu_src)
        x_gpu_dst.set(x_cpu_src)
    with cupy.cuda.stream.Stream() as stream_dtoh:
        # x_gpu_src.get(out=x_pinned_cpu_dst)
        x_cpu_dst = x_gpu_src.get()

    end = stream_htod.record()
    stream_dtoh.synchronize()
    stream_htod.synchronize()

    print('Asynchronous Device to Host / Host to Device (ms)')
    print(cupy.cuda.get_elapsed_time(start, end))


# copy_tensor_torch(10, 4096, True, True)

# demo_cupy()