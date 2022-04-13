import torch
import argparse
from glue_dataset.qqp import get_glue_qqp_train_data_loader
from glue_dataset.tokenizer import build_tokenizer
from modules.gpt_modules import GPTGlueModel, get_position_id
from deepspeed.profiling.flops_profiler import FlopsProfiler
from optimizer.optimizer import get_fp16_optimizer


def main():
    parser = argparse.ArgumentParser(description='Test Glue-qqp dataset')
    parser.add_argument('--train-data', nargs='+', default=['./glue_dataset/data/QQP/train.tsv'], metavar='S',
                        help='path to the training data')
    parser.add_argument('--valid-data', nargs='+', default=['./glue_dataset/data/QQP/test.tsv'], metavar='S',
                        help='path to the training data')
    parser.add_argument('--tokenizer-type', type=str, default='BertWordPieceLowerCase', metavar='S',
                        help='which tokenizer to use.')
    parser.add_argument('--vocab-file', type=str, default='./glue_dataset/data/bert-large-cased-vocab.txt', metavar='S',
                        help='which tokenizer to use.')
    parser.add_argument('--vocab-extra-ids', type=int, default=0, metavar='N',
                        help='-')
    parser.add_argument('--make-vocab-size-divisible-by', type=int, default=128, metavar='N',
                        help='-')
    parser.add_argument('--use-cuda', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='if this is set to True, will use cuda to train')
    parser.add_argument('--cuda-id', type=int, default=0, metavar='N',
                        help='cuda index, if the instance has multiple GPUs.')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--seq-length', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--embedding-dim', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--num-layers', type=int, default=24, metavar='N',
                        help='-')
    parser.add_argument('--num-heads', type=int, default=16, metavar='N',
                        help='-')
    parser.add_argument('--lr', type=float, default=0.01, metavar='N',
                        help='-')
    parser.add_argument('--loss-scale', type=float, default=64,
                        help='Static loss scaling, positive power of 2 values can improve fp16 convergence. ')
    parser.add_argument('--fp16', action='store_true',
                        help='Run model in fp16 mode.')
    args = parser.parse_args()
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')
    if args.fp16:
        print("<=== Training in fp16. ===>")
    tokenizer = build_tokenizer(args)
    print("token vocab size:", tokenizer.vocab_size)
    data_loader = get_glue_qqp_train_data_loader(args, tokenizer)
    num_classes = 2
    model = GPTGlueModel(args, tokenizer.vocab_size, num_classes, use_checkpoint=True).to(device)
    print("Model info:")
    for name, param in model.named_parameters():
        print(name, ":", param.size())

    if args.fp16:
        model.half()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    if args.fp16:
        optimizer = get_fp16_optimizer(args, optimizer)

    prof = FlopsProfiler(model)

    # for i in range(len(data_loader)):
    #    data = data_loader[i]
    # train_data_loader_iter = iter(data_loader)
    # print(next(train_data_loader_iter))
    for i, data in enumerate(data_loader):
        if i == 1:
            prof.start_profile()

        input_ids = data['text'].to(device)
        # position_ids = get_position_id(args.seq_length, args.batch_size, device)
        labels = data['label'].to(device)

        optimizer.zero_grad(set_to_none=True)
        # output = model(input_ids, position_ids)
        output = model(input_ids)
        print(output.shape)
        # loss = loss_func(output, labels)
        loss = torch.nn.functional.cross_entropy(output, labels)

        if i == 1:
            prof.stop_profile()
            flops = prof.get_total_flops()
            macs = prof.get_total_macs()
            params = prof.get_total_params()
            prof.print_model_profile()
            prof.end_profile()
            print("Flops:", flops)
            print("Macs:", macs)
            print("Params:", params)

        loss.backward()
        optimizer.step()

        print("Iter ", i, "===== Loss: ", loss.item(), "======")
        if i > 10:
            break
        # print(data)


if __name__ == '__main__':
    main()
