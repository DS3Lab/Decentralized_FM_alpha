from transformers import pipeline, set_seed

set_seed(32)
generator = pipeline('text-generation', model="xzyao/TXVHF8OJ400A4MK7TIVZGSEQGYP33M594NCREZPIGZ5ZVPD4J9", do_sample=True, num_return_sequences=1)
generator("The woman worked as a")