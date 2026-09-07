import os
import logging
from huggingface_hub import snapshot_download

logger = logging.getLogger("cognifin.rag.assets")

ASSET_MODE = os.getenv("ASSET_MODE", "local")
HF_REPO_ID = os.getenv("HF_REPO_ID", "")


def ensure_index_cache():
    if ASSET_MODE == "local":
        logger.info("Using local assets.")
        return

    if os.path.exists("index_cache/faiss.index"):
        logger.info("index_cache already exists and populated.")
        return

    if not HF_REPO_ID:
        logger.error("action=ensure_index_cache status=missing_config error='HF_REPO_ID environment variable is not configured.'")
        return

    logger.info(f"Downloading index cache from HuggingFace dataset '{HF_REPO_ID}'...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir="index_cache",
        token=os.getenv("HF_TOKEN") or None,
    )
    logger.info("Index cache download complete.")