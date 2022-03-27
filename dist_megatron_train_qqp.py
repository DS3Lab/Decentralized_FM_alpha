import sys
sys.path.append('../Megatron-LM')


# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLUE finetuning/evaluation."""
from functools import partial
import torch
import time
import torch.nn.functional as F
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron import get_timers
from megatron import mpu
from megatron.model import ModelType
from megatron.model.classification import Classification
from tasks.eval_utils import accuracy_func_provider
from megatron.training import pretrain
from megatron.initialize import initialize_megatron
from megatron.utils import average_losses_across_data_parallel_group
from glue_dataset.qqp import QQPDataset
from megatron.training import setup_model_and_optimizer, print_datetime, build_train_valid_test_data_iterators


def train_valid_dataset_provider(train_val_test_num_samples):
    """Build train and validation dataset."""
    args = get_args()
    tokenizer = get_tokenizer()
    train_dataset = QQPDataset('training', args.train_data_path,
                               tokenizer, args.seq_length)
    valid_dataset = QQPDataset('validation', args.valid_data_path,
                               tokenizer, args.seq_length)
    test_dataset = QQPDataset('test', args.test_data_path,
                              tokenizer, args.seq_length)

    return train_dataset, valid_dataset, test_dataset


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    num_classes = 2
    print_rank_0('building classification model for QQP ...')
    model = Classification(num_classes=num_classes, num_tokentypes=2,
                           pre_process=pre_process, post_process=post_process)
    return model


def loss_func(loss_mask, sentence_order, output_tensor):
    lm_loss_, sop_logits = output_tensor

    lm_loss_ = lm_loss_.float()
    loss_mask = loss_mask.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

    if sop_logits is not None:
        sop_loss = F.cross_entropy(sop_logits.view(-1, 2).float(),
                                   sentence_order.view(-1),
                                   ignore_index=-1)
        sop_loss = sop_loss.float()
        loss = lm_loss + sop_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss, sop_loss])
        return loss, {'lm loss': averaged_losses[0],
                      'sop loss': averaged_losses[1]}
    else:
        loss = lm_loss
        averaged_losses = average_losses_across_data_parallel_group(
            [lm_loss])
        return loss, {'lm loss': averaged_losses[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    keys = ['text', 'types', 'labels', 'is_random', 'loss_mask', 'padding_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)
    tokens = data_b['text'].long()
    types = data_b['types'].long()
    sentence_order = data_b['is_random'].long()
    loss_mask = data_b['loss_mask'].float()
    lm_labels = data_b['labels'].long()
    padding_mask = data_b['padding_mask'].long()
    timers('batch-generator').stop()

    if not args.bert_binary_head:
        types = None

    # Forward pass through the model.
    output_tensor = model(tokens, padding_mask, tokentype_ids=types,
                          lm_labels=lm_labels)

    return output_tensor, partial(loss_func, loss_mask, sentence_order)


def train_qqp(train_valid_dataset_provider,
              model_provider,
              model_type,
              forward_step_func,
              extra_args_provider=None,
              args_defaults={}):
    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    _TRAIN_START_TIME = time.time()
    start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()
    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')

    args = get_args()
    timers = get_timers()

    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup').start()

    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider,
                                                               model_type)
    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')

    # Data stuff.
    timers('train/valid-data-iterators-setup').start()
    print_datetime('after dataloaders are built')
    train_data_iterator, valid_data_iterator, test_data_iterator = build_train_valid_test_data_iterators(
        train_valid_dataset_provider)
    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup', 'train/valid-data-iterators-setup'])
    print_rank_0('training ...')

    iteration = 0

if __name__ == '__main__':
    """Finetune/evaluate."""
    print("Start training.")
    train_qqp(train_valid_dataset_provider, model_provider,
              ModelType.encoder_or_decoder, forward_step)
