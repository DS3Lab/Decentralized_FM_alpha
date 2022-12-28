import os
import logging
from typing import Dict
from together_worker.fast_training import FastTrainingInterface
from together_web3.together import TogetherWeb3, TogetherClientOptions
import time
import random
import string
import subprocess
import os
import traceback
class FinetuneOPT(FastTrainingInterface):
    def __init__(self, model_name: str, args=None) -> None:
        args = args if args is not None else {}
        super().__init__(model_name, args)

    def dispatch_request(self, args, env) -> Dict:
        match_event = env
        try:
            project_id = "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(50))
            # call the finetune script
            my_env = os.environ.copy()
            my_env['PROJECT_ID'] = project_id
            my_env['DATASET_URL'] = args[0]['dataset_url']
            for k in args[0]['arguments']:
                my_env[k.upper()] = str(args[0]['arguments'][k])
            if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
                my_env['CUDA_VISIBLE_DEVICES']="0,1"
            proc = subprocess.Popen(
                "/bin/bash api_finetune.sh",
                shell=True,
                env=my_env
            )
            self.update_step({
                "step": 0,
                "finetune_id": project_id,
            }, match_event)
            proc.wait()
            return {"finetune_id": project_id}
        except Exception as e:
            traceback.print_exc()

if __name__ == "__main__":
    coord_url = os.environ.get("COORD_URL", "localhost")
    coordinator = TogetherWeb3(
        TogetherClientOptions(reconnect=True),
        http_url=os.environ.get("COORD_HTTP_URL", f"http://{coord_url}:8094"),
        websocket_url=os.environ.get("COORD_WS_URL", f"ws://{coord_url}:8093/websocket"),
    )
    fip = FinetuneOPT(model_name=os.environ.get("MODEL", "FT_OPT1.3B"), args={
        "auth_token": os.environ.get("AUTH_TOKEN"),
        "coordinator": coordinator,
        "device": os.environ.get("DEVICE", "cuda"),
        "gpu_num": 0,
        "gpu_type": 'titanxp',
        "stream_tokens_pipe": True,
        "gpu_mem": 12,
        "group_name": os.environ.get("GROUP", "group1"),
        "worker_name": os.environ.get("WORKER", "worker1"),
    })
    fip.start()