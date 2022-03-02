
from ..tasks.squad import SQuAD2


class SquadDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def _get_dataset(task, tokenizer, data_split='train', data_style='bert'):
    if data_split == 'train':
        data = task.training_docs()
    elif data_split == 'validation':
        data = task.validation_docs()
    elif data_split == 'test':
        data = task.test_docs()
    else:
        raise Exception('Unrecognized data split.')
        
    contexts = []
    questions = []
    answers = []
    for qa in data:
        context = qa['context']
        question = qa['question']
        for answer_text, answer_start  in zip(qa['answers']['text'], qa['answers']['answer_start']):
            answer = {'text': answer_text, 'answer_start': answer_start,}
            contexts.append(context)
            questions.append(question)
            answers.append(answer)
                
    # add end idx to answers
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters
    
    if tokenizer.pad_token is None:
        print('Setting pad_token to eos_token...')
        tokenizer.pad_token = tokenizer.eos_token
    
    if data_style == 'bert':
    
        encodings = tokenizer(contexts, questions, truncation=True, padding=True)

        start_positions = []
        end_positions = []
        for i in range(len(answers)):
            start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
            end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
            # if None, the answer passage has been truncated
            if start_positions[-1] is None:
                start_positions[-1] = tokenizer.model_max_length
            if end_positions[-1] is None:
                end_positions[-1] = tokenizer.model_max_length
        encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
        
    elif data_style == 'gpt':
        
        pass
    
    else:
        raise Exception('Unrecognized data style.')
    
    dataset = SquadDataset(encodings)
    
    return dataset


def get_squad2_train_data_loader(args, tokenizer, num_workers=0):
    task = SQuAD2()
    train_dataset = _get_dataset(task=task, tokenizer=tokenizer, data_split='train', data_style='bert')
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size,
                                                    sampler=train_sampler,
                                                    shuffle=False,
                                                    num_workers=num_workers,
                                                    drop_last=True,
                                                    pin_memory=True,
                                                    collate_fn=None)
    return train_data_loader