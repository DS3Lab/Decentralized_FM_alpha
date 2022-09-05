import json
import argparse
import os
from filelock import SoftFileLock
import netifaces as ni
import requests


def define_nccl_port_by_job_id(job_id: int):
    return 10000 + job_id % 3571  # make sure different job use different port


class CoordinatorInferenceHTTPClient:
    def __init__(self, args, model_name: str) -> None:
        self.working_directory = args.working_directory
        self.job_id = args.job_id
        self.model_name = model_name
        self.dir_path = os.path.join(self.working_directory, self.model_name)
        lock_path = os.path.join(self.dir_path, self.model_name + '.lock')
        self.model_lock = SoftFileLock(lock_path, timeout=10)

    def notify_inference_heartbeat(self):
        pass

    def notify_inference_join(self):
        ip = ni.ifaddresses('access')[ni.AF_INET][0]['addr']
        return requests.post("http://coordinator.shift.ml/eth/rank/"+str(self.job_id),
                             json={"ip": ip, "rank": 0}).json()

    def load_input_job_from_dfs(self, job_id):
        doc_path = os.path.join(self.dir_path, 'input_' + job_id + '.json')
        if os.path.exists(doc_path):
            with self.model_lock:
                with open(doc_path, 'r') as infile:
                    doc = json.load(infile)
            return doc
        else:
            return None

    def save_output_job_to_dfs(self, result_doc):
        output_filename = 'output_' + result_doc['_id'] + '.json'
        output_path = os.path.join(self.dir_path, output_filename)
        with self.model_lock:
            with open(output_path, 'w') as outfile:
                json.dump(result_doc, outfile)
        input_filename = 'input_' + result_doc['_id'] + '.json'
        input_path = os.path.join(self.dir_path, input_filename)
        assert os.path.exists(input_path)
        os.remove(input_path)



