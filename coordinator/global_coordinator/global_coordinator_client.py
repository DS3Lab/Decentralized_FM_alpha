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

    def put_request_cluster_coordinator(self, request_doc: dict, inference_result) -> dict:
        print("=========put_request_cluster_coordinator=========")
        print(request_doc)
        request_doc['inference_result'] = inference_result
        request_doc['job_state'] = 'job_finished'
        request_doc = self.db.save(request_doc)
        print(f"=========[cluster client] put result in key value store=========")
        print(request_doc)
        print("-----------------------------------------------------------------")

    def get_request_cluster_coordinator(self) -> dict:
        print("=========get_request_cluster_coordinator=========")
        for doc in self.db.all():
            # print(doc)
            # print('job_type_info' in doc['doc'])
            doc = doc['doc']
            if 'job_type_info' in doc and doc['job_type_info'] == 'latency_inference':
                if doc['job_state'] == 'job_queued':
                    doc['job_state'] = 'job_running'
                    doc = self.db.save(doc)
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
    parser.add_argument('--op', type=str, default='put', metavar='S',
                        help='The op: {get or put}.')
    args = parser.parse_args()
    print(vars(args))
    client = GlobalCoordinatorClient(args)

    if args.op == 'get':
        client.get_request_cluster_coordinator()
    elif args.op == 'put':
        inference_results = ["Hello world reply."]
        req_doc={

        }
        client.put_request_cluster_coordinator(req_doc, inference_results)
    else:
        assert False


if __name__ == '__main__':
    main()