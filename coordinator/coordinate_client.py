import socket
import argparse


def client_message_parser(msg: bytes, context: str):
    msg_arg = msg.decode().split('#')
    if context == 'join_training':
        arg_dict = {'prime_ip': msg_arg[0],
                    'my_rank': int(msg_arg[1])}
    else:
        assert False
    return arg_dict


# The client port should be determined by the job-id which is unique. The ip + port will identify a worker.
class CoordinatorClient:
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


def main():
    parser = argparse.ArgumentParser(description='Test Coordinator-Client')
    parser.add_argument('--coordinator-server-port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--lsf-job-no', type=str, default='100', metavar='S',
                        help='Job-<ID> assigned by LSF.')
    args = parser.parse_args()
    print(vars(args))
    client = CoordinatorClient(args)
    client.notify_train_join()
    client.notify_train_finish("0#6.88")


if __name__ == '__main__':
    main()
