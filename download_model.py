import os

from huggingface_hub import snapshot_download
from transformers.utils import move_cache

MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
model_name = MODEL_ID.split("/")[-1]
MODEL_DIR = os.path.expanduser(os.path.join("~", "models", model_name))

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # using safetensors 
    )
    move_cache()


if __name__ == "__main__":
    main()
