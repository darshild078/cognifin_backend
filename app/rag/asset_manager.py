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


def extract_cache(tar_path: str):
    logger.info(f"Extracting index cache from {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=".")
    logger.info("Index cache extraction complete.")


def ensure_index_cache():
    if ASSET_MODE == "local":
        logger.info("Using local assets.")
        return

    if os.path.exists("index_cache") and os.path.exists("index_cache/faiss.index"):
        logger.info("index_cache already exists and populated.")
        return

    tar_path = download_cache()
    extract_cache(tar_path)