from comm.comm_utils import *


def distributed_inference_foo_iter(args, pipeline, device, infer_data_loader):
    total_time = 0
    if get_pipeline_parallel_rank() == 0:
        for i, data in enumerate(infer_data_loader):
            input_ids = data['text'].to(device)
            current_iter_time = pipeline.inference_batch(input_ids)
            
            '''
            save generated tokens
            '''
            if pipeline.pp_rank == 0:
                tokenizer = infer_data_loader.dataset.tokenizer
                generate_ids = torch.cat(pipeline.recv_new_token, 1)
                print('A sample:')
                print('prompt:', tokenizer.decode(
                    input_ids[0]
                ).replace('<|endoftext|>', ''))
                print('generate:', tokenizer.decode(
                    generate_ids[0]
                ).replace('<|endoftext|>', ''))
            
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters-1:
                break
        averaged_time = total_time / (args.num_iters - 1)
        print("Finished running ", args.num_iters,
              " iterations, averaged (exclude the first iter) run time:", averaged_time)
    else:
        i = 0
        while True:
            current_iter_time = pipeline.inference_batch(None)
            if i > 0:
                total_time += current_iter_time
            if i >= args.num_iters - 1:
                break
            i += 1
        averaged_time = total_time / (args.num_iters - 1)
    return averaged_time
