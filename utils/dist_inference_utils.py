import os
import json
from datetime import datetime
from comm.comm_utils import *


def distributed_inference_foo_iter(args, pipeline, device, request_processor):
    
    total_time = 0
    if get_pipeline_parallel_rank() == 0:
        output_requests = []
        infer_data_loader = request_processor.get_dataloader(args.batch_size)
        for i, inputs in enumerate(infer_data_loader):
            input_ids = inputs['text'].to(device)
            output_ids_list = []
            current_iter_time = pipeline.inference_batch(input_ids, output_ids_list)
            request_processor.add_result(inputs, output_ids_list)
            
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters-1:
                break
        averaged_time = total_time / (args.num_iters - 1 + 1e-9)
        print("Finished running ", args.num_iters,
              " iterations, averaged (exclude the first iter) run time:", averaged_time)
        
        request_processor.write_scenario_state()
            
    else:
        i = 0
        while True:
            current_iter_time = pipeline.inference_batch(None)
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters - 1:
                break
            i += 1
        averaged_time = total_time / (args.num_iters - 1 + 1e-9)
    return averaged_time


def distributed_inference_mask_iter(args, pipeline, device, request_processor):
    
    total_time = 0
    if get_pipeline_parallel_rank() == 0:
        output_requests = []
        infer_data_loader = request_processor.get_dataloader(args.batch_size)
        for i, inputs in enumerate(infer_data_loader):
            input_ids = inputs['text'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            output_ids_list = []
            current_iter_time = pipeline.inference_batch(input_ids, output_ids_list, attention_mask=attention_mask)
            request_processor.add_result(inputs, output_ids_list, batch_time=current_iter_time)
            
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters-1:
                break
        averaged_time = total_time / (args.num_iters - 1 + 1e-9)
        print("Finished running ", args.num_iters,
              " iterations, averaged (exclude the first iter) run time:", averaged_time)
        
        request_processor.write_scenario_state()
            
    else:
        infer_data_loader = request_processor.get_dataloader(args.batch_size)
        for i, inputs in enumerate(infer_data_loader):
            attention_mask = inputs['attention_mask'].to(device)
            current_iter_time = pipeline.inference_batch(attention_mask=attention_mask)
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters-1:
                break
        averaged_time = total_time / (args.num_iters - 1 + 1e-9)
    return averaged_time
