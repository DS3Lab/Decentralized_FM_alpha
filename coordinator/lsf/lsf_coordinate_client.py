import json
import argparse
import os
from filelock import SoftFileLock
import netifaces as ni
import requests
import time


def define_nccl_port_by_job_id(job_id: int):
    return 10000 + job_id % 3571  # make sure different job use different port


def alias_to_model_name(model_alias: str) -> str:
    # print(torch_type)

    mappings = {
        'stable_diffusion': 'stable_diffusion',
        'Image: stable_diffusion': 'stable_diffusion',
        'gpt_j_6B': 'gpt_j_6b',
        'gpt-j-6B': 'gpt_j_6b',
        'EleutherAI/gpt-j-6B': 'gpt_j_6b',
        'gpt-neox-20b-new': 'gpt_neox_20b',
        'T0pp-new': 't0_pp_11b',
        't5-11b-new': 't5_11b',
        'ul2-new': 'ul2',
        'opt_66B': 'opt_66b',
        'opt-66b-new': 'opt_66b',
        'opt-175b-new': 'opt_175b',
        'bloom-new': 'bloom',
        'yalm-100b-new': 'yalm',
        'glm-130b-new': 'glm',
        'multimodalart/latentdiffusion': None
    }
    return mappings[model_alias]


class CoordinatorInferenceHTTPClient:
    def __init__(self, args, model_name: str) -> None:
        self.working_directory = args.working_directory
        self.job_id = args.job_id
        self.model_name = model_name
        # self.dir_path = os.path.join(self.working_directory, self.model_name)
        self.dir_path = os.path.join(self.working_directory)
        lock_path = os.path.join(self.dir_path, self.model_name + '.lock')
        self.model_lock = SoftFileLock(lock_path, timeout=10)

    def notify_inference_heartbeat(self):
        pass

    def notify_inference_join(self, netname='access'):
        ip = ni.ifaddresses(netname)[ni.AF_INET][0]['addr']
        return requests.post("http://coordinator.shift.ml/eth/rank/"+str(self.job_id),
                             json={"ip": ip}).json()

    def update_status(self, new_status, returned_payload=None):
        return requests.post(f"https://coordinator.shift.ml/eth/update_status/{self.job_id}", json={
            "status": new_status,
            "returned_payload": returned_payload,
            "timestamp": time.time()
        })

    def load_input_job_from_dfs(self, job_id, return_path=False):
        doc_path = os.path.join(self.dir_path, 'input_' + job_id + '.json')
        print("<load_input_job_from_dfs - doc_path>:", doc_path)
        if return_path:
            if os.path.exists(doc_path):
                return doc_path
            else:
                print("Warning none input file found!!!!!!!!!!")
                return None
        else:
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



