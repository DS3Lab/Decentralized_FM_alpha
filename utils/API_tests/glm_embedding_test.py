import requests


my_post_dict = {
    "type": "general",
    "payload": {
        "max_tokens": 32,
        "n": 1,
        "temperature": 0.8,
        "top_p": 0.6,
        "top_k": 5,
        "model": "glm",
        #"prompt": ["Where is Zurich?"],
        #"prompt":  [' '.join(["1" for _ in range(128)]) for _ in range(48)],
        "prompt": ["Where is Zurich?",
                   "Where is LA's airport?",
		 		   "Where is Houston?",
                   "Where is Austin's train station?"],
        "request_type": "language-model-inference",
        "stop": [],
        "best_of": 1,
        "logprobs": 0,
        "echo": False,
        "prompt_embedding": False
    },
    "returned_payload": {},
    "status": "submitted",
    "source": "dalle",
}


res = requests.post("https://planetd.shift.ml/jobs", json=my_post_dict).json()

print(res)
