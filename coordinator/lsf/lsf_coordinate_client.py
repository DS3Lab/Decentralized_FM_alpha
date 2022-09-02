import json
import socket
import argparse
import os


def client_message_parser(msg: bytes, context: str):
    msg_arg = msg.decode().split('#')
    if context == 'join_training':
        arg_dict = {'prime_ip': msg_arg[0],
                    'my_rank': int(msg_arg[1])}
    elif context == 'join_inference':
        arg_dict = {'prime_ip': msg_arg[0],
                    'my_rank': int(msg_arg[1]),
                    'port': int(msg_arg[2])}
    else:
        assert False
    return arg_dict


# The client port should be determined by the job-id which is unique. The ip + port will identify a worker.
class CoordinatorTrainClient:
    def __init__(self, args):
        self.host_ip = args.coordinator_server_ip
        self.host_port = args.coordinator_server_port
        self.client_port = int(args.lsf_job_no) % 10000 + 10000

    def notify_train_join(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            s.sendall(b"train#join")
            msg = s.recv(1024)
            print(f"Received: {msg}")
            msg_arg = client_message_parser(msg, 'join_training')
            return msg_arg['prime_ip'], msg_arg['my_rank']

    def notify_train_finish(self, message: str):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            s.sendall(b"train#finish#"+message.encode())
            msg = s.recv(1024)
            print(f"Received: {msg}")


class CoordinatorInferenceClient:
    def __init__(self, args):
        self.host_ip = args.coordinator_server_ip
        self.host_port = args.coordinator_server_port
        self.client_port = int(args.lsf_job_no) % 10000 + 10000
        self.working_directory = args.working_directory

    def notify_inference_join(self):
        print("++++++++++++++++++notify_inference_join++++++++++++++++++")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            # s.sendall(b"inference#join#")
            msg_dict = {
                'task': 'inference',
                'state': 'join'
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(1024)
            print(f"Received: {msg}")
            msg_arg = client_message_parser(msg, 'join_inference')
            return msg_arg['prime_ip'], msg_arg['my_rank'], msg_arg['port']

    def notify_inference_finish(self, rank: int, iter_time: float):
        print("++++++++++++++++++notify_inference_finish++++++++++++++++++")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            # s.sendall(b"inference#finish#"+message.encode())
            msg_dict = {
                'task': 'inference',
                'state': 'finish',
                'rank': rank,
                'iter_time': iter_time
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(1024)
        print(f"Received: {msg}")
        return msg

    def notify_inference_heartbeat(self):
        print("++++++++++++++++++notify_inference_heartbeat++++++++++++++++++")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            # s.sendall(b"inference#finish#"+message.encode())
            msg_dict = {
                'task': 'inference',
                'state': 'worker_heartbeats',
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(1024)
        print(f"Received: {msg}")
        return msg

    def load_input_job_from_dfs(self, model_name: str):
        print("++++++++++++++load_input_job_from_dfs++++++++++++")
        dir_path = os.path.join(self.working_directory, model_name)
        for filename in os.listdir(dir_path):
            print(filename)
            if filename.startswith('input_'):
                doc_path = os.path.join(dir_path, filename)
                with open(doc_path, 'r') as infile:
                    doc = json.load(infile)
                    # assert model_name == doc['task_api']['model_name']
                    return doc
        return None

    def save_output_job_to_dfs(self, model_name, result_doc):
        print("++++++++++++++save_output_job_to_dfs++++++++++++")
        dir_path = os.path.join(self.working_directory, model_name)
        output_filename = 'output_' + result_doc['_doc'] + '.json'
        output_path = os.path.join(dir_path, output_filename)
        with open(output_path, 'w') as outfile:
            json.dump(result_doc, outfile)
        input_filename = 'input_' + result_doc['_doc'] + '.json'
        input_path = os.path.join(dir_path, input_filename)
        assert os.path.exists(input_path)
        os.remove(input_path)

    """
    def notify_inference_dequeue_job(self, model_name):
        print("++++++++++++++++++notify_inference_dequeue_job++++++++++++++++++")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            # s.sendall(b"inference#finish#"+message.encode())
            msg_dict = {
                'task': 'inference',
                'state': 'worker_dequeue',
                'model_name': model_name
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(8192)
            return_msg_dict = json.loads(msg)
        print(f"Received: {return_msg_dict}")
        return return_msg_dict

    def notify_inference_post_result(self, job_request):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            # s.sendall(b"inference#finish#"+message.encode())
            msg_dict = {
                'task': 'inference',
                'state': 'worker_post_result',
                'job_request': job_request
            }
            s.sendall(json.dumps(msg_dict).encode())
            msg = s.recv(8192)
        print(f"Received: {msg}")
        return msg
    """


class CoordinatorHybridInferenceClient:
    def __init__(self, args):
        self.host_ip = args.coordinator_server_ip
        self.host_port = args.coordinator_server_port
        self.client_port = int(args.lsf_job_no) % 10000 + 10000
        self.node_type = args.node_type

    def notify_inference_join(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            s.sendall(b"inference#join#"+self.node_type.encode())
            msg = s.recv(1024)
            print(f"Received: {msg}")
            msg_arg = client_message_parser(msg, 'join_inference')
            return msg_arg['prime_ip'], msg_arg['my_rank'], msg_arg['port']

    def notify_inference_finish(self, message: str):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            s.sendall(b"inference#finish#"+message.encode())
            msg = s.recv(1024)
            print(f"Received: {msg}")


def main():
    parser = argparse.ArgumentParser(description='Test Coordinator-Client')
    parser.add_argument('--coordinator-type', type=str, default='train', help='train or inference')
    parser.add_argument('--coordinator-server-port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--lsf-job-no', type=str, default='100', metavar='S',
                        help='Job-<ID> assigned by LSF.')
    args = parser.parse_args()
    print(vars(args))
    print(vars(args))
    if args.coordinator_type == 'train':
        client = CoordinatorTrainClient(args)
        client.notify_train_join()
        client.notify_train_finish("0#6.88")
    elif args.coordinator_type == 'inference':
        client = CoordinatorInferenceClient(args)
        client.notify_inference_join()
        client.notify_inference_finish("0#6.88")
    else:
        assert False


if __name__ == '__main__':
    main()
