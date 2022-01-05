import torch
import argparse
from glue_dataset.qqp import QQPDataset
from glue_dataset.tokenizer import build_tokenizer

from dist_gpt_module import GPTEmbedding, GPTTransformerLayer, GlueClassification, get_position_id


class GPTLocaModel(torch.nn.Module):
    def __init__(self, args, vocab_size, num_classes):
        super(GPTLocaModel, self).__init__()
        self.embedding = GPTEmbedding(vocab_size, args.embedding_dim, args.seq_length)

        module_list = []
        for _ in range(args.num_layers):
            module_list.append(GPTTransformerLayer(args.embedding_dim, args.num_heads, args.embedding_dim*4))
        self.transformers = torch.nn.Sequential(*module_list)
        self.classifier = GlueClassification(args.embedding_dim, num_classes)

    def forward(self, input_ids, position_ids):
        input_emb = self.embedding(input_ids, position_ids)
        output_emb = self.transformers(input_emb)
        return self.classifier(output_emb)


def train_data_loader(args, tokenizer):
    train_dataset = QQPDataset('training', args.train_data, tokenizer, args.seq_length)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.macro_batch_size,
                                                    sampler=train_sampler,
                                                    shuffle=False,
                                                    num_workers=4,
                                                    drop_last=True,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader


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
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--macro-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--seq-length', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--embedding-dim', type=int, default=768, metavar='N',
                        help='-')
    parser.add_argument('--num-layers', type=int, default=2, metavar='N',
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
    data_loader = train_data_loader(args, tokenizer)
    num_classes = 2
    model = GPTLocaModel(args, tokenizer.vocab_size, num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    for data in data_loader:
        input_ids = data['text'].to(device)
        position_ids = get_position_id(args.seq_length, args.macro_batch_size).to(device)
        labels = data['label'].to(device)

        optimizer.zero_grad()
        output = model(input_ids, position_ids)
        print(output.shape)
        # loss = loss_func(output, labels)
        loss = torch.nn.functional.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()
        print("===== Loss: ", loss.item(), "======")
        # print(data)



if __name__ == '__main__':
    main()