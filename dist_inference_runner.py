import argparse
import torch.autograd.profiler as profiler
from utils.dist_args_utils import *
from utils.dist_inference_utils import *
from comm.comm_utils import *
from pipeline_parallel.dist_pipeline_inference_greedy import DistGreedyInferenceAsync
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description='Inference Runner')
    add_device_arguments(parser)
    add_torch_distributed_arguments(parser)
    add_model_arguments(parser)
    add_qqp_task_arguments(parser)
    add_training_hyper_parameter_arguments(parser)
    add_mixed_precision_arguments(parser)
    add_parallel_schema_arguments(parser)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model-name', type=str, default='', metavar='S',
                        help='trained model path')
    parser.add_argument('--profiling', type=str, default='tidy_profiling', metavar='S',
                        help='enable which profiling? default: tidy mode')
    parser.add_argument('--trace-postfix', type=str, default='default', metavar='S',
                        help='postfix of the tracing file name.')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')

    init_communicators(args)

    if get_pipeline_parallel_rank() == 0:
        
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.model_max_length = args.input_seq_length

        ### TODO some dataset for inference.
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.data = tokenizer(['you are']*128, return_tensors='pt')

            def __len__(self):
                return len(self.data['input_ids'])

            def __getitem__(self, idx):
                if torch.is_tensor(idx):
                    idx = idx.tolist()

                item = {
                    k: v[idx] for k, v in self.data.items()
                }
                item['text'] = item['input_ids']

                return item

        dataset = CustomDataset()
        infer_data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size
        )
        
    else:
        infer_data_loader = None

    pipe = DistGreedyInferenceAsync(args, device)

    if args.profiling == 'no-profiling':
        distributed_inference_foo_iter(args, pipe, device, infer_data_loader)
    else:
        prefix = './trace_json/gpt3_' + args.pp_mode
        trace_file = prefix + get_learning_arguments_str(args) + get_model_arguments_str(args) + \
                     get_dist_arguments_str(args) + get_mixed_precision_arguments_str(args) + '_' + \
                     args.profiling + '_' + args.trace_postfix + '.json'
        if args.profiling == 'tidy_profiling':
            distributed_inference_foo_iter(args, pipe, device, infer_data_loader)
            pipe.export_profiling_result(filename=trace_file)
        elif args.profiling == 'pytorch_profiling':
            with profiler.profile(profile_memory=True, use_cuda=args.use_cuda) as prof:
                distributed_inference_foo_iter(args, pipe, device, infer_data_loader)
            print(prof.key_averages().table())
            prof.export_chrome_trace(trace_file)
        else:
            print("No recognized profiler?")
            assert False


if __name__ == '__main__':
    main()
