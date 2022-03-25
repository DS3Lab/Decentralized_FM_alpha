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
from megatron.utils import average_losses_across_data_parallel_group


def glue_classification(num_classes, Dataset,
                        name_from_datapath_func):

    def train_valid_datasets_provider():
        """Build train and validation dataset."""
        args = get_args()
        tokenizer = get_tokenizer()

        train_dataset = Dataset('training', args.train_data,
                                tokenizer, args.seq_length)
        valid_dataset = Dataset('validation', args.valid_data,
                                tokenizer, args.seq_length)

        return train_dataset, valid_dataset

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""
        args = get_args()

        print_rank_0('building classification model for {} ...'.format(
            args.task))
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

    def metrics_func_provider():
        """Privde metrics callback function."""
        def single_dataset_provider(datapath):
            args = get_args()
            tokenizer = get_tokenizer()

            name = name_from_datapath_func(datapath)
            return Dataset(name, [datapath], tokenizer, args.seq_length)
        return accuracy_func_provider(single_dataset_provider)

    """Finetune/evaluate."""
    pretrain(train_valid_datasets_provider, model_provider,
             ModelType.encoder_or_decoder, forward_step)


def main():
    args = get_args()

    assert args.task == 'QQP'
    num_classes = 2
    from tasks.glue.qqp import QQPDataset as Dataset

    def name_from_datapath(datapath):
        return datapath.split('QQP')[-1].strip('.tsv').strip('/').replace('_', '-')

    glue_classification(num_classes, Dataset, name_from_datapath)