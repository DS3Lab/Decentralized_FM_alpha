import deepspeed
import argparse
import time
from glue_dataset.qqp import QQPDataset
from glue_dataset.tokenizer import build_tokenizer
from utils.dist_gpt_utils import *
from modules.gpt_modules import GPTGlueModel, get_position_id


def main():
    parser = argparse.ArgumentParser(description='ZeRO-GPT3')
    add_model_arguments(parser)
    add_task_arguments(parser)
    add_training_hyper_parameter_arguments(parser)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    tokenizer = build_tokenizer(args)
    print("token vocab size:", tokenizer.vocab_size)
    train_dataset = QQPDataset('training', args.train_data, tokenizer, args.seq_length)

    num_classes = 2
    model = GPTGlueModel(args, tokenizer.vocab_size, num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(args=args, model=model, optimizer=optimizer,
                                                                        model_parameters=model.parameters(),
                                                                        training_data=train_dataset)
    for i, data in enumerate(train_dataloader):
        start_time = time.time()
        input_ids = data['text']
        position_ids = get_position_id(args.seq_length, args.batch_size, None)
        labels = data['label']
        output = model_engine(input_ids, position_ids)
        loss = torch.nn.functional.cross_entropy(output, labels)
        forward_time = time.time()
        print("Forward pass takes {:3.2f}s".format(forward_time - start_time))
        model_engine.backward(loss)
        backward_time = time.time()
        print("Backward pass takes {:3.2f}s".format(backward_time - forward_time))
        model_engine.step()
        end_time = time.time()
        print("Whole iteration takes {:3.2f}s".format(end_time - start_time))
        # print(data)
        if i >= args.num_iters - 1:
            break


if __name__ == '__main__':
    main()
