import os
import json
import argparse

# TODO: change this hard coded version later.
node_ip_lists = [
    "34.212.121.230",
    "34.221.72.170",
    "54.187.136.136",
]


def download_trace_logs(args, prefix, ips=node_ip_lists):
    if os.path.isdir('./'+prefix):
        os.rmdir('./'+prefix)
    else:
        os.mkdir('./'+prefix)
    for i in range(args.world_size):
        os.system("scp -i ../binhang_ds3_aws_oregon.pem ubuntu@" + ips[i] +
                  ":~/GPT-home-private/trace_json/"+prefix+'_'+str(i)+'.json ./' + prefix)


def merge_logs(args, prefix):
    result = []
    for i in range(args.world_size):
        print(i)
        with open("./" + prefix + '/' + prefix + '_' + str(i) + ".json") as inputJson:
            current_trace = json.load(inputJson)
            inputJson.close()
            for log in current_trace:
                log['pid'] = 'Node ' + str(i)
            result.extend(current_trace)
    print(len(result))
    with open("./" + prefix + ".json", 'w') as outputJson:
        json.dump(result, outputJson)


def main():
    parser = argparse.ArgumentParser(description='Gpipe-GPT3')
    parser.add_argument('--world-size', type=int, default=4, metavar='D',
                        help='world-size (default: 2)')
    parser.add_argument('--batch-size', type=int, default=12, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--embedding-dim', type=int, default=768, metavar='N',
                        help='-')
    parser.add_argument('--seq-length', type=int, default=2048, metavar='N',
                        help='-')
    args = parser.parse_args()
    prefix = "gpt3_gpipe_b" + str(args.batch_size) + "_l" + str(args.seq_length) + '_m' + str(args.embedding_dim) + \
             "_w" + str(args.world_size)
    download_trace_logs(args, prefix)
    merge_logs(args, prefix)


if __name__ == '__main__':
    main()