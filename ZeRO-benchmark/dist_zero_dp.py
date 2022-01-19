import deepspeed
import argparse
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
        input_ids = data['text']
        position_ids = get_position_id(args.seq_length, args.batch_size, None)
        labels = data['label']
        output = model_engine(input_ids, position_ids)
        print(output.shape)
        loss = torch.nn.functional.cross_entropy(output, labels)
        model_engine.backward(loss)
        model_engine.step()
        print("Iter ", i, "===== Loss: ", loss.item(), "======")
        # print(data)


if __name__ == '__main__':
    main()
