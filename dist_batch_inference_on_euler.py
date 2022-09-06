import argparse
from pipeline_parallel.dist_pp_utils import get_pp_inference_module
from utils.dist_args_utils import *
from utils.dist_inference_utils import *
from comm.comm_utils import *
from coordinator.lsf.lsf_coordinate_client import CoordinatorInferenceHTTPClient
from coordinator.lsf.lsf_job_scheduler import alias_to_model_name
from task_datasets.inference_data import get_request_processor


def sync_setting(args, pipeline, device, return_msg=None):
    num_return_sequences_tensor = torch.zeros(1, dtype=torch.int64, device=device)
    generate_token_length_tensor = torch.zeros(1, dtype=torch.int64, device=device)
    temperature_tensor = torch.zeros(1, dtype=torch.float32, device=device)
    top_p_tensor = torch.zeros(1, dtype=torch.float32, device=device)
    do_sample_tensor = torch.zeros(1, dtype=torch.uint8, device=device)

    if get_pipeline_parallel_rank() == 0:
        generate_token_length = return_msg['task_api']['parameters']['max_new_tokens']
        do_sample = return_msg['task_api']['parameters']['do_sample']
        temperature = return_msg['task_api']['parameters']['temperature']
        top_p = return_msg['task_api']['parameters']['top_p']
        num_return_sequences = return_msg['task_api']['parameters']['num_return_sequences']
        num_return_sequences_tensor[:] = num_return_sequences
        generate_token_length_tensor[:] = generate_token_length
        temperature_tensor[:] = temperature
        top_p_tensor[:] = top_p
        do_sample_tensor[:] = do_sample

    pipeline.comm.broadcast(num_return_sequences_tensor, src=0)
    pipeline.num_completions = num_return_sequences_tensor.item()

    pipeline.comm.broadcast(generate_token_length_tensor, src=0)
    pipeline.generate_seq_length = generate_token_length_tensor.item()

    pipeline.comm.broadcast(temperature_tensor, src=0)
    args.temperature = temperature_tensor.item()

    pipeline.comm.broadcast(top_p_tensor, src=0)
    args.top_p = top_p_tensor.item()

    pipeline.comm.broadcast(do_sample_tensor, src=0)
    if do_sample_tensor.item() == 0:
        args.temperature = 0

    pipeline.change_buffer_size()


def main():
    parser = argparse.ArgumentParser(description='Inference Runner with coordinator.')
    add_device_arguments(parser)
    add_torch_distributed_inference_w_euler_coordinator_arguments(parser)
    add_inference_arguments(parser)
    add_inference_details_arguments(parser)
    add_global_coordinator_arguments(parser)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--job_id', type=str, default='-', metavar='S',
                        help='DB ID')
    parser.add_argument('--profiling', type=str, default='tidy_profiling', metavar='S',
                        help='enable which profiling? default: tidy mode')
    parser.add_argument('--trace-postfix', type=str, default='default', metavar='S',
                        help='postfix of the tracing file name.')
    args = parser.parse_args()
    print_arguments(args)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')

    model_name_abbr = args.model_name.split('/')[-1]
    print("model name abbr: ", model_name_abbr)
    print("model name: ", alias_to_model_name(model_name_abbr))
    coord_client = CoordinatorInferenceHTTPClient(args, alias_to_model_name(model_name_abbr))

    res = coord_client.notify_inference_join()
    prime_ip = res['prime_ip']
    rank = res['rank']
    port = res['nccl_port']

    print("<====Coordinator assigned prime-IP:", prime_ip, " and my assigned rank", rank, "====>")

    init_inference_communicators_with_coordinator(args, prime_ip, rank, port=port)

    input_path = coord_client.load_input_job_from_dfs(args.job_id, return_path=True)
    request_processor = get_request_processor(args, infer_data=input_path)
    request_processor.set_arguments(args)

    pipe = get_pp_inference_module(args, device, rank=rank)

    tokenizer = get_tokenizer(args)
    tokenizer.model_max_length = args.input_seq_length

    print(f"Inference pipeline loading model <{model_name_abbr}> is done!")
    if get_pipeline_parallel_rank() == 0:
        coord_client.update_status("running")

    if args.profiling == 'no-profiling':
        avg_iter_time = distributed_inference_mask_iter(args, pipe, device, request_processor)
    else:
        prefix = './trace_json/inference_' + args.pp_mode
        trace_file = prefix + get_inference_arguments_str(args, rank=rank) + '_' + args.profiling + '_' + \
                     args.trace_postfix + '.json'
        if args.profiling == 'tidy_profiling':
            avg_iter_time = distributed_inference_mask_iter(args, pipe, device, request_processor)
            pipe.export_profiling_result(filename=trace_file)
        else:
            print("No recognized profiler?")
            assert False
    if get_pipeline_parallel_rank() == 0:
        coord_client.update_status("finished", returned_payload=request_processor.data)


if __name__ == '__main__':
    main()
