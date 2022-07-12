import argparse
import torch.autograd.profiler as profiler
from utils.dist_args_utils import *
from utils.dist_inference_utils import *
from comm.comm_utils import *
from task_datasets.inference_data import get_inference_data_loader
from pipeline_parallel.dist_pp_utils import *
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description='Inference Runner')
    add_device_arguments(parser)
    add_torch_distributed_arguments(parser)
    # add_model_arguments(parser)
    # add_qqp_task_arguments(parser)
    add_inference_arguments(parser)
    # add_training_hyper_parameter_arguments(parser)
    # add_mixed_precision_arguments(parser)
    # add_parallel_schema_arguments(parser)
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model-name', type=str, default='./pretrained_models/gpt2', metavar='S',
                        help='trained model path')
    parser.add_argument('--model-type', type=str, default='gpt2', metavar='S',
                        help='trained model path')
    parser.add_argument('--infer-data', type=str, default='', metavar='S',
                        help='data path')
    parser.add_argument('--top-k', type=int, default=1, metavar='S',
                        help='sample from top k')
    parser.add_argument('--top-p', type=float, default=1, metavar='S',
                        help='sample from top p')
    parser.add_argument('--temperature', type=float, default=1, metavar='S',
                        help='temperature on logits')
    # TODO: trivial
    parser.add_argument('--echo-prompt', type=lambda x: (str(x).lower() == 'true'),
                        default=False, metavar='S',
                        help='append prompt to the generated text')
    # TODO
    parser.add_argument('--num-completions', type=int, default=1, metavar='S',
                        help='num of completions')
    # TODO
    parser.add_argument('--top-k-per-token', type=int, default=1, metavar='S',
                        help='return top k candidate for each token')
    parser.add_argument('--profiling', type=str, default='tidy_profiling', metavar='S',
                        help='enable which profiling? default: tidy mode')
    parser.add_argument('--trace-postfix', type=str, default='default', metavar='S',
                        help='postfix of the tracing file name.')
    parser.add_argument('--fp16', action='store_true',
                        help='Run model in fp16 mode.')
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
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
        tokenizer.model_max_length = args.input_seq_length

        ### TODO some dataset for inference.
        if args.infer_data != '':
            infer_data_loader = get_inference_data_loader(args, tokenizer, num_workers=0)
        else:
            class CustomDataset(torch.utils.data.Dataset):
                def __init__(self, tokenizer):
                    self.tokenizer = tokenizer
                    self.data = self.tokenizer(
                        [('you are').strip()]*1280, 
                        return_tensors='pt', padding='max_length', truncation=True,
                    )

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

            dataset = CustomDataset(tokenizer)
            infer_data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size
            )
        
    else:
        tokenizer = None
        infer_data_loader = None

    pipe = get_pp_inference_module(args, device)

    if args.profiling == 'no-profiling':
        distributed_inference_foo_iter(args, pipe, device, infer_data_loader)
    else:
        prefix = './trace_json/inference_' + args.pp_mode
        trace_file = prefix + get_inference_arguments_str(args) + args.profiling + '_' + args.trace_postfix + '.json'
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
