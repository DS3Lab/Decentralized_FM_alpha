import socket
import argparse
from collections import OrderedDict
import os


def server_message_parser(msg: bytes):
    msg_arg = msg.decode().split('#')
    arg_dict = {'task': msg_arg[0],
                'state': msg_arg[1]}
    if arg_dict['task'] == 'train':
        if arg_dict['state'] == 'finish':
            arg_dict['rank'] = int(msg_arg[2])
            arg_dict['iter_time'] = float(msg_arg[3])
        elif arg_dict['state'] == 'submit':
            arg_dict['job_name'] = msg_arg[2]
    return arg_dict


class CoordinatorServer:
    def __init__(self, args):
        self.host = args.coordinator_server_ip
        self.port = args.coordinator_server_port
        # An array of dict object to store worker info
        self.worker_nodes = OrderedDict()
        self.prime_worker_ip = None
        self.bsub_script_path = args.bsub_script_path
        self.train_demand_workers = 0

    def _get_rank0_ip(self):
        assert len(self.worker_nodes) > 0 and self.prime_worker is not None
        return self.prime_worker_ip

    def _print_current_working_nodes(self):
        print("<----------------Current Working Nodes---------------->")
        for node_key in self.worker_nodes.keys():
            print(f"Node rank {self.worker_nodes[node_key]['rank']}, Address: {node_key}")
        print("-------------------------------------------------------")

    def _handle_train_submit(self, job_name) -> str:
        print("<<<<<<<<<<<<<<<<<<<<< Submit Job >>>>>>>>>>>>>>>>>>>>>>")
        if self.train_demand_workers != 0:
            return 'Current training task is still running, cannot handle your job submission.'
        else:
            if job_name == 'lsf_gpt3xl_3gpu':
                self.train_demand_workers = 3
            elif job_name == 'lsf_gpt3xl_64gpu':
                self.train_demand_workers = 64
            else:
                return f'This job is not recognized on coordinate - {job_name}'
            for i in range(self.train_demand_workers):
                os.system(f"rm {self.bsub_script_path}/submit_cache/*.bsub")
                os.system(f"cp {self.bsub_script_path}/{job_name}.bsub "
                          f"{self.bsub_script_path}/submit_cache/{job_name}_{i+1}.bsub")
                os.system(f"echo \' {i+1}\' >> {self.bsub_script_path}/submit_cache/{job_name}_{i+1}.bsub")
                os.system(f"cd {self.bsub_script_path}/submit_cache && "
                          f"bsub < {job_name}_{i+1}.bsub")
            os.system("bjobs")
            return f'Succeed to submit job - {job_name}'

    def _handle_train_join(self, worker_ip, port) -> str:
        node_key = worker_ip + ':' + str(port)
        assert node_key not in self.worker_nodes, f"Worker called notify_train_join has been joined before ({node_key})"
        print(f"Connected by +NEW+ worker with address {worker_ip}, (port:{port})")
        new_node_rank = len(self.worker_nodes)
        if new_node_rank == 0:
            self.prime_worker_ip = worker_ip
        self.worker_nodes[node_key] = {'rank': new_node_rank}
        return_msg = self.prime_worker_ip + '#' + str(self.worker_nodes[node_key]['rank'])
        return return_msg

    def _handle_train_finish(self, worker_ip, port, msg_arg) -> str:
        node_key = worker_ip + ':' + str(port)
        assert node_key in self.worker_nodes, f"Worker called notify_train_finish is not recognized ({node_key})"
        print(f"Connected by known worker with address {worker_ip}, (port:{port}), allocated rank "
              f"{self.worker_nodes[node_key]['rank']}")
        print(f"<=====Training finished on rank-{msg_arg['rank']} worker, "
              f"average time {msg_arg['iter_time']} seconds.=====>")
        del self.worker_nodes[node_key]
        self.train_demand_workers -= 1
        return_msg = 'done'
        return return_msg

    def execute_server(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            while True:
                connection, address = s.accept()
                with connection:
                    worker_ip, port = address
                    msg_data = connection.recv(1024)
                    print(f"==[Recv message: {msg_data}]==")
                    msg_arg = server_message_parser(msg_data)
                    if msg_arg['task'] == 'train':
                        if msg_arg['state'] == 'submit':
                            return_msg = self._handle_train_submit(msg_arg['job_name'])
                        elif msg_arg['state'] == 'join':
                            return_msg = self._handle_train_join(worker_ip, port)
                        elif msg_arg['state'] == 'finish':
                            return_msg = self._handle_train_finish(worker_ip, port, msg_arg)
                        else:
                            assert False, f"Not valid operator for training ({msg_arg['state']})"
                    connection.sendall(return_msg.encode())
                    connection.close()
                    self._print_current_working_nodes()


def main():
    parser = argparse.ArgumentParser(description='Test Coordinator-Server')
    parser.add_argument('--coordinator-server-port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--bsub-script-path', type=str,
                        default='/cluster/home/biyuan/GPT-home-private/scripts/lsf_scripts', metavar='S',
                        help='Path to store the bsub scripts')
    args = parser.parse_args()
    print(vars(args))
    coordinator = CoordinatorServer(args)
    coordinator.execute_server()


if __name__ == '__main__':
    main()
