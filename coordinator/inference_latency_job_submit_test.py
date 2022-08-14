import time
import socket
import argparse
import random
import json


class JobSubmitClient:
    def __init__(self, args):
        self.host_ip = args.coordinator_server_ip
        self.host_port = args.coordinator_server_port
        self.client_port = 9999 - random.randint(1, 5000)  # cannot exceed 10000

    def submit_inference_task(self, job_details: dict):
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', self.client_port))
                s.connect((self.host_ip, self.host_port))
                s.sendall(b"inference#latency_job#" + json.dumps(job_details).encode())
                msg = s.recv(1024)
                print(f"Received: {msg}")


def main():
    parser = argparse.ArgumentParser(description='Test Job-Submit-Client')
    parser.add_argument('--submit-job', type=str, default='inference', help='train or inference')
    parser.add_argument('--coordinator-server-port', type=int, default=9002, metavar='N',
                        help='The port of coordinator-server.')
    parser.add_argument('--coordinator-server-ip', type=str, default='localhost', metavar='S',
                        help='The IP of coordinator-server.')
    parser.add_argument('--job-name', type=str, default='lsf_gptJ_inf_4RTX2080Ti', metavar='S',
                        help='Support a fixed list of job first, this can be more flexible later.')
    args = parser.parse_args()
    print(vars(args))
    client = JobSubmitClient(args)
    myobj = {
        'inputs': "you are not",
        "parameters": {
            "max_new_tokens": 20, "return_full_text": True,
            "do_sample": True, "temperature": 0.8, "top_p": 0.95,
            "max_time": 10.0, "num_return_sequences": 2,
            "use_gpu": True,
        }
    }
    print(json.dumps(myobj).encode())
    client.submit_inference_task(myobj)


if __name__ == '__main__':
    main()
