import socket
import argparse
from collections import OrderedDict


def server_message_parser(msg: bytes):
    msg_arg = msg.decode().split('#')
    arg_dict = {'task': msg_arg[0],
                'state': msg_arg[1]}
    if arg_dict['task'] == 'train' and arg_dict['state'] == 'finish':
        arg_dict['rank'] = int(msg_arg[2])
        arg_dict['iter_time'] = float(msg_arg[3])
    return arg_dict


class CoordinatorServer:
    def __init__(self, args):
        self.host = args.coordinator_server_ip
        self.port = args.port
        # An array of dict object to store worker info
        self.worker_nodes = OrderedDict()
        self.prime_worker_ip = None

    def _get_rank0_ip(self):
        assert len(self.worker_nodes) > 0 and self.prime_worker is not None
        return self.prime_worker_ip

    def _print_current_working_nodes(self):
        print("<===============Current Working Nodes===============>")
        for node_key in self.worker_nodes.keys():
            print(f"Node rank {self.worker_nodes[node_key]['rank']}, Address: {node_key}")

    def execute_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            while True:
                connection, address = s.accept()
                with connection:
                    worker_ip, port = address
                    node_key = worker_ip + str(port)
                    if node_key not in self.worker_nodes:
                        print(f"Connected by +NEW+ worker with address {worker_ip}, (port:{port})")
                        new_node_rank = len(self.worker_nodes)
                        if new_node_rank == 0:
                            self.prime_worker_ip = worker_ip
                        self.worker_nodes[node_key] = {'rank': new_node_rank}
                    else:
                        print(f"Connected by known worker with address {worker_ip}, (port:{port}), allocated rank "
                              f"{self.worker_nodes[worker_ip]['rank']}")
                    msg_data = connection.recv(1024)
                    print(f"Recv message: {msg_data}")
                    msg_arg = server_message_parser(msg_data)
                    if msg_arg['task'] == 'train':
                        if msg_arg['state'] == 'join':
                            return_msg = self.prime_worker_ip + '#' + str(self.worker_nodes[node_key]['rank'])
                        elif msg_arg['state'] == 'finish':
                            return_msg = 'done'
                            print(f"<=====Training finished on rank-{msg_arg['rank']} worker, "
                                  f"average time {msg_arg['iter_time']} seconds.=====>")
                            del self.worker_nodes[node_key]
                    connection.sendall(return_msg.encode())
                    connection.close()
                    self._print_current_working_nodes()


def main():
    parser = argparse.ArgumentParser(description='Test Coordinator-Server')
    parser.add_argument('--port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    args = parser.parse_args()
    coordinator = CoordinatorServer(args)
    coordinator.execute_server()


if __name__ == '__main__':
    main()
