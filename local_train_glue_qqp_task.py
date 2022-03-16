import torch
import argparse
from glue_dataset.qqp import get_glue_qqp_train_data_loader
from glue_dataset.tokenizer import build_tokenizer
from modules.gpt_modules import GPTGlueModel, get_position_id


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
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--seq-length', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--embedding-dim', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--num-layers', type=int, default=16, metavar='N',
                        help='-')
    parser.add_argument('--num-heads', type=int, default=16, metavar='N',
                        help='-')
    parser.add_argument('--lr', type=float, default=0.01, metavar='N',
                        help='-')
    args = parser.parse_args()
    if args.use_cuda:
        assert (torch.cuda.is_available())
        device = torch.device('cuda', args.cuda_id)
    else:
        device = torch.device('cpu')
    tokenizer = build_tokenizer(args)
    print("token vocab size:", tokenizer.vocab_size)
    data_loader = get_glue_qqp_train_data_loader(args, tokenizer)
    num_classes = 2
    model = GPTGlueModel(args, tokenizer.vocab_size, num_classes, use_checkpoint=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # for i in range(len(data_loader)):
    #    data = data_loader[i]
    # train_data_loader_iter = iter(data_loader)
    # print(next(train_data_loader_iter))
    for i, data in enumerate(data_loader):

        input_ids = data['text'].to(device)
        # position_ids = get_position_id(args.seq_length, args.batch_size, device)
        labels = data['label'].to(device)

        optimizer.zero_grad(set_to_none=True)
        # output = model(input_ids, position_ids)
        output = model(input_ids)
        print(output.shape)
        # loss = loss_func(output, labels)
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        print("Iter ", i, "===== Loss: ", loss.item(), "======")
        if i > 20:
            break
        # print(data)


if __name__ == '__main__':
    main()
