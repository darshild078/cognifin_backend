import os
import tarfile
import logging
from huggingface_hub import hf_hub_download

logger = logging.getLogger("cognifin.rag.assets")

ASSET_MODE = os.getenv("ASSET_MODE", "local")


def download_cache():
    repo_id = os.getenv("HF_REPO_ID", "darshild078/finsightai-assets")
    filename = os.getenv("HF_CACHE_FILENAME", "index_cache.tar.gz")
    
    logger.info(f"Downloading index cache from HuggingFace repo: {repo_id} via SDK...")
    tar_path = hf_hub_download(repo_id=repo_id, filename=filename)
    logger.info("Download complete via Hugging Face SDK.")
    return tar_path


def extract_cache():

    print("Extracting index cache...")

    with tarfile.open("index_cache.tar.gz", "r:gz") as tar:
        tar.extractall()

    os.remove("index_cache.tar.gz")

    print("Extraction complete.")


def ensure_index_cache():

    if ASSET_MODE == "local":
        print("Using local assets.")
        return

    if os.path.exists("index_cache"):
        print("index_cache already exists.")
        return

    download_cache()
    extract_cache()