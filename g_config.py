data_dir = None
host = "http://127.0.0.1:5000/"

model = "gpt-4"
#model = "gpt-3.5-turbo"
#model = "gpt-3.5-turbo-16k"

openai_key = "SET YOUR OPENAI KEY HERE"

max_tokens = 4096

# Override any of the above settings with values from g_local_config
try:
    from g_local_config import *
except ModuleNotFoundError:
    pass
