import os
from huggingface_hub import login

def hfh_login():
    # Read token from HFH_TOKEN environment variable and fail with a helpful message if missing
    token = os.getenv("HFH_TOKEN")
    if token:
        login(token=token)
    else:
        print("HFH_TOKEN environment variable is not set. Please set it to your Hugging Face token if you need to access private models.")
