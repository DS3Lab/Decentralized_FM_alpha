from comm.comm_utils import *
from modules.hf_gpt2_train_module import gpt_loss_func

from itertools import tee, islice, chain


def distributed_train_foo_iter(args, pipeline, device, train_data_loader):
    total_time = 0
    if get_pipeline_parallel_rank() == 0:
        for i, data in enumerate(train_data_loader):
            input_ids = data['text'].to(device)
            current_iter_time = pipeline.sgd_iter(input_ids, None)
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters-1:
                break
        averaged_time = total_time / (args.num_iters - 1)
        print("Finished running ", args.num_iters,
              " iterations, averaged (exclude the first iter) run time:", averaged_time)
    elif get_pipeline_parallel_rank() == args.pipeline_group_size - 1:
        for i, data in enumerate(train_data_loader):
            if args.task == 'SeqClassification':
                labels = data['label'].to(device)
            elif args.task == 'Seq2SeqClassification':
                labels = data['text'].to(device)
            else:
                print("Not supported task!")
                assert False
            current_iter_time = pipeline.sgd_iter(None, labels)
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters-1:
                break
        averaged_time = total_time / (args.num_iters - 1)
    else:
        i = 0
        while True:
            current_iter_time=pipeline.sgd_iter(None, None)
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters-1:
                break
            i += 1
        averaged_time = total_time / (args.num_iters - 1)
    return averaged_time



def distributed_train_lm_iter(args, pipeline, device, train_data_loader):
    pipeline.model.train() # Flag .training to True to enable Dropout
    if get_pipeline_parallel_rank() == 0:
        total_time = 0
        for i, data in enumerate(train_data_loader):
            input_ids = data['text'].to(device)
            current_iter_time = pipeline.sgd_iter(input_ids, None)
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters-1:
                break
        averaged_time = total_time / (args.num_iters - 1)
        print("Finished running ", args.num_iters,
              " iterations, averaged (exclude the first iter) run time:", averaged_time)
    elif get_pipeline_parallel_rank()  == args.pipeline_group_size - 1:
        for i, data in enumerate(train_data_loader):
            input_ids = data['text'].to(device)
            labels = data['text'].to(device) # labels are inputs
            pipeline.sgd_iter(input_ids, labels, 
                              loss_func=gpt_loss_func) # lm loss func
            if i >= args.num_iters-1:
                break
    else:
        for i, data in enumerate(train_data_loader):
            pipeline.sgd_iter(None, None)
            i += 1
            if i >= args.num_iters:
                    break