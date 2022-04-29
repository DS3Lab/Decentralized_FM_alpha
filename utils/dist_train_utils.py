from comm.comm_utils import *


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