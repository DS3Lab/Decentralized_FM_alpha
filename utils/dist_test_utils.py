from comm.comm_utils import *
import datasets

def distributed_test_foo_iter(args, pipeline, device, test_data_loader):
    pipeline.model.eval()
    if get_pipeline_parallel_rank() == 0:
        for i, data in enumerate(test_data_loader):
            input_ids = data['text'].to(device)
            current_iter_time = pipeline.infer_iter(input_ids, None, None)
    elif get_pipeline_parallel_rank()  == args.pipeline_group_size - 1:
        metric = datasets.load_metric('accuracy')
        for i, data in enumerate(test_data_loader):
            input_ids = data['text'].to(device)
            labels = data['label'].to(device)
            data_ids = data['idx']
            pipeline.infer_iter(input_ids, labels, data_ids, metric=metric)
        print(metric.compute())
    else:
        for i, data in enumerate(test_data_loader):
            pipeline.infer_iter(None, None, None)