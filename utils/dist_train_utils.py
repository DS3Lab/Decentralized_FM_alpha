from comm.comm_utils import *
from modules.gpt_modules import gpt_loss_func


def distributed_train_foo_iter(args, pipeline, device, train_data_loader):
    pipeline.model.train() # Flag .training to True to enable Dropout
    if get_pipeline_parallel_rank() == 0:
        total_time = 0
        for i, data in enumerate(train_data_loader):
            input_ids = data['text'].to(device)
            data_ids = data['idx']
            current_iter_time = pipeline.sgd_iter(input_ids, None, data_ids)
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
            labels = data['label'].to(device)
            data_ids = data['idx']
            pipeline.sgd_iter(input_ids, labels, data_ids)
            if i >= args.num_iters-1:
                break
    else:
        for i, data in enumerate(train_data_loader):
            data_ids = data['idx']
            pipeline.sgd_iter(None, None, data_ids)
            i += 1
            if i >= args.num_iters:
                    break
                    
                    
                    
def distributed_train_lm_iter(args, pipeline, device, train_data_loader):
    pipeline.model.train() # Flag .training to True to enable Dropout
    if get_pipeline_parallel_rank() == 0:
        total_time = 0
        for i, data in enumerate(train_data_loader):
            input_ids = data['text'].to(device)
            data_ids = data['idx']
            current_iter_time = pipeline.sgd_iter(input_ids, None, data_ids)
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
            data_ids = data['idx']
            pipeline.sgd_iter(input_ids, labels, data_ids, 
                              loss_func=gpt_loss_func) # lm loss func
            if i >= args.num_iters-1:
                break
    else:
        for i, data in enumerate(train_data_loader):
            data_ids = data['idx']
            pipeline.sgd_iter(None, None, data_ids)
            i += 1
            if i >= args.num_iters:
                    break
                    
                    
                    
                    
def distributed_train_bert_iter(args, pipeline, device, train_data_loader):
    pipeline.model.train() # Flag .training to True to enable Dropout
    if get_pipeline_parallel_rank() == 0:
        total_time = 0
        for i, data in enumerate(train_data_loader):
            inputs_ids = data['text'].to(device)
            aux_inputs = {
                'token_type_ids': data['token_type_ids'].to(device),
                'attention_mask': data['attention_mask'].to(device),
            }
            data_ids = data['idx']
            current_iter_time = pipeline.sgd_iter(inputs_ids, None, data_ids, aux_input_data=aux_inputs)
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters-1:
                break
        averaged_time = total_time / (args.num_iters - 1)
        print("Finished running ", args.num_iters,
              " iterations, averaged (exclude the first iter) run time:", averaged_time)
    elif get_pipeline_parallel_rank()  == args.pipeline_group_size - 1:
        for i, data in enumerate(train_data_loader):
            aux_inputs = {
                'attention_mask': data['attention_mask'].to(device),
            }
            input_ids = data['text'].to(device)
            labels = data['label'].to(device)
            data_ids = data['idx']
            pipeline.sgd_iter(None, labels, data_ids, aux_input_data=aux_inputs)
            if i >= args.num_iters-1:
                break
    else:
        for i, data in enumerate(train_data_loader):
            aux_inputs = {
                'attention_mask': data['attention_mask'].to(device),
            }
            data_ids = data['idx']
            pipeline.sgd_iter(None, None, data_ids, aux_input_data=aux_inputs)
            i += 1
            if i >= args.num_iters:
                    break