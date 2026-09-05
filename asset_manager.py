import os
import tarfile
import requests

ASSET_MODE = os.getenv("ASSET_MODE", "local")


def download_cache():

    cache_url = os.getenv("HF_CACHE_URL")

    if not cache_url:
        raise ValueError("HF_CACHE_URL is not set")

    print("Downloading index cache from HuggingFace...")

    response = requests.get(cache_url, stream=True)

    response.raise_for_status()

    with open("index_cache.tar.gz", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Download complete.")


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