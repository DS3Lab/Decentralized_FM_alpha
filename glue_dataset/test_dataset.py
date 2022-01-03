import argparse
from qqp import QQPDataset
# This is different from Megatron
from torchtext.data.utils import get_tokenizer


def train_valid_datasets_provider(args):
    """Build train and validation dataset."""
    tokenizer = get_tokenizer('basic_english')

    train_dataset = QQPDataset('training', args.train_data,
                            tokenizer, args.seq_length)
    valid_dataset = QQPDataset('validation', args.valid_data,
                            tokenizer, args.seq_length)

    return train_dataset, valid_dataset


def main():
    parser = argparse.ArgumentParser(description='Test Glue-qqp dataset')
    parser.add_argument('--train-data', nargs='+', default=['./data/QQP/train.tsv'], metavar='S',
                        help='path to the training data')
    parser.add_argument('--valid-data', nargs='+', default=['./data/QQP/test.tsv'], metavar='S',
                        help='path to the training data')
    parser.add_argument('--seq-length', type=int, default=2048, metavar='N',
                        help='-')
    args = parser.parse_args()
    train_valid_datasets_provider(args)


if __name__ == '__main__':
    main()