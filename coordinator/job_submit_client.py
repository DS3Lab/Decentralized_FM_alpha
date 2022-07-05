import socket
import argparse


class JobSubmitClient:
    def __init__(self, args):
        self.host_ip = args.coordinator_server_ip
        self.host_port = args.coordinator_server_port
        self.client_port = 9999

    def submit_train_job(self, job_name: str):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.client_port))
            s.connect((self.host_ip, self.host_port))
            s.sendall(b"train#submit#"+job_name)
            msg = s.recv(1024)
            print(f"Received: {msg}")


def main():
    parser = argparse.ArgumentParser(description='Test Job-Submit-Client')
    parser.add_argument('--coordinator-server-port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--job-name', type=str, default='lsf_gpt3small_1gpu_3node.bsub', metavar='S',
                        help='Support a fixed list of job first, this can be more flexible later.')
    args = parser.parse_args()
    print(vars(args))
    client = JobSubmitClient(args)
    client.submit_train_job(args.job_name)


if __name__ == '__main__':
    main()
