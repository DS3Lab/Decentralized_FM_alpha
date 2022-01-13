import os
import json
import argparse

# TODO: change this hard coded version later.
node_ip_lists = [
    "35.88.99.51",
    "54.185.191.6",
    "54.218.168.226",
]


def download_trace_logs(args, prefix, postfix, ips=node_ip_lists):
    if os.path.isdir('./'+prefix):
        os.rmdir('./'+prefix)
    else:
        os.mkdir('./'+prefix)
    for i in range(args.world_size):
        os.system("scp -i ../binhang_ds3_aws_oregon.pem ubuntu@" + ips[i] +
                  ":~/GPT-home-private/trace_json/"+prefix+'_'+str(i) + postfix + ' ./' + prefix)


def merge_logs(args, prefix, postfix):
    result = []
    current_min_stamp = float('inf')
    for i in range(args.world_size):
        print(i)
        with open("./" + prefix + '/' + prefix + '_' + str(i) + postfix) as inputJson:
            current_trace = json.load(inputJson)
            inputJson.close()
            if i == 0:
                for log in current_trace:
                    current_min_stamp = min(log['ts'], current_min_stamp)
            for log in current_trace:
                log['pid'] = 'Node ' + str(i)
                log['ts'] = log['ts'] - current_min_stamp
            result.extend(current_trace)
    print(len(result))
    with open("./" + prefix + postfix, 'w') as outputJson:
        json.dump(result, outputJson)


def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT3')
    parser.add_argument('--world-size', type=int, default=3, metavar='D',
                        help='world-size (default: 2)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--micro-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--embedding-dim', type=int, default=768, metavar='N',
                        help='-')
    parser.add_argument('--seq-length', type=int, default=2048, metavar='N',
                        help='-')
    parser.add_argument('--profiling', type=str, default='tidy_profiling', metavar='S',
                        help='enable which profiling? default: tidy mode')
    args = parser.parse_args()
    prefix = 'gpt3_gpipe_b' + str(args.batch_size) + '_' + str(args.micro_batch_size) + '_l' + str(args.seq_length) + \
             '_m' + str(args.embedding_dim) + '_w' + str(args.world_size)
    postfix = '_' + args.profiling + '.json'
    download_trace_logs(args, prefix, postfix)
    merge_logs(args, prefix, postfix)


if __name__ == '__main__':
    main()
