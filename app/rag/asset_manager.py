from pathlib import Path
from huggingface_hub import snapshot_download

from app.core.config import settings
from app.core.logging import logger


def ensure_index_cache() -> None:
    index_path = Path(settings.INDEX_CACHE_DIR) / "faiss.index"
    if index_path.exists():
        logger.info("module=assets action=ensure_cache status=skipped reason=cache_exists")
        return

    if not settings.HF_REPO_ID:
        logger.error("module=assets action=ensure_cache status=failed error='HF_REPO_ID is not configured'")
        return

    logger.info(f"module=assets action=download_cache repo_id={settings.HF_REPO_ID} status=starting")
    try:
        snapshot_download(
            repo_id=settings.HF_REPO_ID,
            repo_type="dataset",
            local_dir=settings.INDEX_CACHE_DIR,
            token=settings.HF_TOKEN or None,
        )
        logger.info(f"module=assets action=download_cache repo_id={settings.HF_REPO_ID} status=success")
    except Exception as exc:
        logger.error(f"module=assets action=download_cache repo_id={settings.HF_REPO_ID} status=failed error='{exc}'")
        raise
