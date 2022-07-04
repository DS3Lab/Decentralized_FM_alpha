import socket
import argparse


def client_message_parser(msg: bytes, context: str):
    msg_arg = msg.decode().split('#')
    if context == 'join_training':
        arg_dict = {'prime_ip': msg_arg[0],
                    'my_rank': msg_arg[1]}
    else:
        assert False
    return arg_dict


class CoordinatorClient:
    def __init__(self, args):
        self.host = args.coordinator_server_ip
        self.port = args.port

    def train_join(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            s.sendall(b"train#join")
            msg = s.recv(1024)
            print(f"Received: {msg}")
            msg_arg = client_message_parser(msg, 'join_training')
            return msg_arg['prime_ip'], msg_arg['my_rank']


def main():
    parser = argparse.ArgumentParser(description='Test Coordinator-Client')
    parser.add_argument('--port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    args = parser.parse_args()
    client = CoordinatorClient(args)
    client.train_join()


if __name__ == '__main__':
    main()