from datetime import datetime
import argparse
import pycouchdb


class GlobalUserClient:
    def __init__(self, args):
        server = pycouchdb.Server(args.db_server_address)
        self.db = server.database("global_coordinator")
        self.task_keys = []

    def put_request_user_client(self, inference_details: dict):
        print("=========put_request_user_client=========")
        msg_dict = {
            'job_type_info': 'latency_inference',
            'job_state': 'job_queued',
            'time': {
                'job_queued_time': str(datetime.now()),
                'job_start_time': None,
                'job_end_time': None,
                'job_returned_time': None
            },
            'task_api': inference_details
        }
        doc = self.db.save(msg_dict)
        current_job_key = doc['_id']
        self.task_keys.append(current_job_key)
        print(f"=========[user client] put result in key value store=========")
        print("Current key:", current_job_key)
        print(doc)
        print("--------------------------------------------------")

    def get_request_user_client(self, request_key: str):
        print("=========get_request_user_client=========")
        doc = self.db.get(request_key)
        assert doc is not None
        print(f"=========[user client] get result in key value store=========")
        print(doc)
        print("------------------------------------------------------")
        if doc['job_state'] == 'job_finished':
            doc['job_state'] = 'job_returned'
            self.db.save(doc)
        return doc


def main():
    parser = argparse.ArgumentParser(description='Test Job-Submit-Client')
    parser.add_argument('--db-server-address', type=str,
                        default="http://xzyao:agway-fondly-ell-hammer-flattered-coconut@db.yao.sh:5984/", metavar='N',
                        help='Key value store address.')
    parser.add_argument('--op', type=str, default='get', metavar='S',
                        help='The op: {get or put}.')
    parser.add_argument('--request-key', type=str, default="6070cb8cfa50434192e060ed40c9a92e", metavar='N',
                        help='The index of the submitted tasks.')
    parser.add_argument('--inputs', type=str, default='Hello world!', metavar='S',
                        help='The prompt sequence.')
    parser.add_argument('--model-name', type=str, default='gptj', metavar='S',
                        help='-')
    parser.add_argument('--task-type', type=str, default='seq_generation', metavar='S',
                        help='-')
    args = parser.parse_args()
    print(vars(args))
    client = GlobalUserClient(args)

    if args.op == 'get':
        client.get_request_user_client(args.request_key)
    elif args.op == 'put':
        inference_details = {
            'inputs': args.inputs,
            'model_name': args.model_name,
            'task_type': args.task_type,
            "parameters": {
                "max_new_tokens": 64,
                "return_full_text": False,
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 0.95,
                "max_time": 10.0,
                "num_return_sequences": 3,
                "use_gpu": True
            },
            'outputs': None
        }
        client.put_request_user_client(inference_details)
    else:
        assert False


if __name__ == '__main__':
    main()
