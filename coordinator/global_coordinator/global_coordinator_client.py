import time
import socket
import argparse
import random
import json
import pycouchdb


class GlobalCoordinatorClient:
    def __init__(self, args):
        server = pycouchdb.Server(args.db_server_address)
        self.db = server.database("global_coordinator")

    def put_request_cluster_coordinator(self, request_key: str, inference_result) -> dict:
        print("=========put_request_cluster_coordinator=========")
        doc = self.db.get(request_key)
        doc['inference_result'] = inference_result
        self.db.save(doc)
        print(f"=========[cluster client] put result in key value store=========")
        print(doc)
        print("-----------------------------------------------------------------")

    def get_request_cluster_coordinator(self) -> dict:
        print("=========get_request_cluster_coordinator=========")
        for doc in self.db.all():
            # print(doc)
            # print('job_type_info' in doc['doc'])
            if 'job_type_info' in doc['doc'] and doc['doc']['job_type_info'] == 'latency_inference':
                if doc['doc']['job_state'] == 'job_queued':
                    doc['doc']['job_state'] = 'job_running'
                    self.db.save(doc)
                    print(f"=========[cluster client] get task in key value store=========")
                    print(doc)
                    print("---------------------------------------------------------------")
                    return doc
        print(f"=========[cluster client] get task in key value store=========")
        print("None job in the queue")
        print("---------------------------------------------------------------")
        return None


def main():
    parser = argparse.ArgumentParser(description='Test Job-Submit-Client')
    parser.add_argument('--db-server-address', type=str,
                        default="http://xzyao:agway-fondly-ell-hammer-flattered-coconut@db.yao.sh:5984/", metavar='N',
                        help='Key value store address.')
    parser.add_argument('--op', type=str, default='get', metavar='S',
                        help='The op: {get or put}.')
    parser.add_argument('--request-key', type=str, default="6070cb8cfa50434192e060ed40c9a92e", metavar='N',
                        help='The index of the submitted tasks.')
    args = parser.parse_args()
    print(vars(args))
    client = GlobalCoordinatorClient(args)

    if args.op == 'get':
        client.get_request_cluster_coordinator()
    elif args.op == 'put':
        inference_results = "Hello world reply."
        client.put_request_cluster_coordinator(args.task_index, inference_results)
    else:
        assert False


if __name__ == '__main__':
    main()