from transformers import pipeline, AutoModel, OPTForCausalLM

model = OPTForCausalLM.from_pretrained("./model_checkpoints/opt1.3-test/checkpoint_100")

generator = pipeline("text-generation", model=model, tokenizer='gpt2')
result = generator("Hello, I'm am conscious and")
print(result)