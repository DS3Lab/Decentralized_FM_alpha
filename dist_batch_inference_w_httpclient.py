import argparse
from pipeline_parallel.dist_pp_utils import get_pp_inference_module
from utils.dist_args_utils import *
from utils.dist_inference_utils import *
from comm.comm_utils import *
from coordinator.lsf.lsf_coordinate_client import CoordinatorInferenceHTTPClient, alias_to_model_name
from task_datasets.inference_data import get_request_processor


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

    print("Print working directory:", args.working_directory)
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

    if get_pipeline_parallel_rank() == 0:
        coord_client.update_status("running", returned_payload={'state': 'initialized'})

    input_path = coord_client.load_input_job_from_dfs(args.job_id, return_path=True)
    request_processor = get_request_processor(args, infer_data=input_path)
    request_processor.set_arguments(args)

    pipe = get_pp_inference_module(args, device, rank=rank)

    tokenizer = get_tokenizer(args)
    tokenizer.model_max_length = args.input_seq_length

    print(f"Inference pipeline loading model <{model_name_abbr}> is done!")
    if get_pipeline_parallel_rank() == 0:
        coord_client.update_status("running", returned_payload={'state': 'model_loaded'})

    if args.profiling == 'no-profiling':
        avg_iter_time = distributed_inference_mask_iter(args, pipe, device, request_processor, client=coord_client)
    else:
        prefix = './trace_json/inference_' + args.pp_mode
        trace_file = prefix + get_inference_arguments_str(args, rank=rank) + '_' + args.profiling + '_' + \
                     args.trace_postfix + '.json'
        if args.profiling == 'tidy_profiling':
            avg_iter_time = distributed_inference_mask_iter(args, pipe, device, request_processor, client=coord_client)
            pipe.export_profiling_result(filename=trace_file)
        else:
            print("No recognized profiler?")
            assert False
    if get_pipeline_parallel_rank() == get_pipeline_parallel_world_size()-1:
        coord_client.update_status("finished", returned_payload={'result': request_processor.data})


if __name__ == '__main__':
    main()