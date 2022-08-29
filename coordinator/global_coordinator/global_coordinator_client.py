import argparse
import pycouchdb
from datetime import datetime


class GlobalCoordinatorClient:
    def __init__(self, args):
        server = pycouchdb.Server(args.db_server_address)
        self.db = server.database("global_coordinator")

    def put_request_cluster_coordinator(self, request_doc: dict, inference_result) -> dict:
        print("=========put_request_cluster_coordinator=========")
        # print(request_doc)
        request_doc['time']['job_end_time'] = str(datetime.now()),
        request_doc['task_api']['outputs'] = inference_result
        request_doc['job_state'] = 'job_finished'
        request_doc = self.db.save(request_doc)
        print(f"=========[cluster client] put result in key value store=========")
        # print(request_doc)
        print("-----------------------------------------------------------------")
        return request_doc

    def get_request_cluster_coordinator(self, job_type_info='latency_inference',
                                        model_name='gptj', task_type='seq_generation') -> dict:
        print("=========get_request_cluster_coordinator=========")
        # Note this is a preliminary version for latency based inference, we need to add more functionality here.
        for doc in self.db.all():
            # print(doc)
            # print('job_type_info' in doc['doc'])
            doc = doc['doc']
            if 'job_type_info' in doc and doc['job_type_info'] == job_type_info:
                if doc['task_api']['model_name'] == model_name and doc['task_api']['task_type'] == task_type:
                    if doc['job_state'] == 'job_queued':
                        doc['job_state'] = 'job_running'
                        doc['time']['job_start_time'] = str(datetime.now())
                        doc = self.db.save(doc)
                        print(f"=========[cluster client] get task in key value store=========")
                        # print(doc)
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