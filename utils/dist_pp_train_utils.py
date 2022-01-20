def distributed_train_foo_iter(args, gpipe, device, train_data_loader):
    if args.rank == 0:
        total_time = 0
        for i, data in enumerate(train_data_loader):
            input_ids = data['text'].to(device)
            current_iter_time = gpipe.sgd_iter(input_ids, None)
            total_time += current_iter_time
            if i >= args.num_iters-1:
                break
        averaged_time = total_time / args.num_iters
        print("Finished running ", args.num_iters, " iterations, averaged run time:", averaged_time)
    elif args.rank == args.world_size - 1:
        for i, data in enumerate(train_data_loader):
            labels = data['label'].to(device)
            gpipe.sgd_iter(None, labels)
            if i >= args.num_iters-1:
                break
    else:
        i = 0
        while True:
            gpipe.sgd_iter(None, None)
            i += 1
            if i >= args.num_iters:
                break