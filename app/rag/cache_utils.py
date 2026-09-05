"""
FinSight AI - Cache Utilities
================================
Atomic write helpers and cache validation utilities for the
persistent index layer.

Provides crash-safe file operations so the index cache
behaves like a reliable database snapshot.

All temporary files use the `.finsight_tmp_` prefix to avoid
false invalidation from editor/OS temp files.

Author: FinSight AI Team
Stage: 1 Addendum (Persistence Safety)
"""

import os
import json
import glob


# =============================================================================
# CONSTANTS
# =============================================================================

TMP_PREFIX = ".finsight_tmp_"


# =============================================================================
# ATOMIC WRITE OPERATIONS
# =============================================================================

def atomic_write_bytes(path: str, data: bytes) -> None:
    """
    Write bytes to a file atomically.

    Pattern: write → tmp file → fsync → rename to final path.
    If the process crashes mid-write, only the tmp file is affected.

    Args:
        path: Final destination path
        data: Bytes to write
    """
    tmp_path = _tmp_path_for(path)

    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, path)


def atomic_write_json(path: str, obj: dict) -> None:
    """
    Write a JSON-serializable dict to a file atomically.

    Args:
        path: Final destination path
        obj: Dictionary to serialize
    """
    data = json.dumps(obj, indent=2).encode("utf-8")
    atomic_write_bytes(path, data)


def atomic_faiss_write(index, path: str) -> None:
    """
    Write a FAISS index to disk atomically with post-write sanity check.

    Steps:
        1. Write index to tmp file via faiss.write_index()
        2. Sanity check: file size >= 1 KB per 100 vectors
        3. Sanity check: can read last 64 bytes successfully
        4. Rename tmp → final

    If sanity checks fail, tmp is deleted and RuntimeError is raised.

    Args:
        index: FAISS index object
        path: Final destination path
    """
    import faiss

    tmp_path = _tmp_path_for(path)

    # Write to tmp
    faiss.write_index(index, tmp_path)

    # --- Post-write sanity checks ---
    file_size = os.path.getsize(tmp_path)

    # Check 1: Minimum size threshold (1 KB per 100 vectors, minimum 1 KB)
    min_expected = max(1024, (index.ntotal // 100) * 1024)
    if file_size < min_expected:
        os.remove(tmp_path)
        raise RuntimeError(
            f"FAISS index file too small: {file_size} bytes "
            f"(expected >= {min_expected} for {index.ntotal} vectors)"
        )

    # Check 2: Can read last 64 bytes (file not truncated)
    try:
        with open(tmp_path, "rb") as f:
            f.seek(-64, 2)  # Seek 64 bytes from end
            tail = f.read(64)
            if len(tail) != 64:
                raise RuntimeError("Could not read 64-byte tail")
    except Exception as e:
        os.remove(tmp_path)
        raise RuntimeError(f"FAISS index tail-read failed: {e}")

    # Sanity checks passed — atomic rename
    os.replace(tmp_path, path)


# =============================================================================
# TEMP FILE DETECTION
# =============================================================================

def has_leftover_tmp(directory: str) -> bool:
    """
    Check if any scoped temporary files exist in the cache directory.

    Only detects files with the `.finsight_tmp_` prefix — ignores
    editor/OS temp files to prevent false invalidation.

    Args:
        directory: Path to the cache directory

    Returns:
        True if leftover tmp files found (indicates interrupted write)
    """
    if not os.path.isdir(directory):
        return False

    pattern = os.path.join(directory, f"{TMP_PREFIX}*")
    matches = glob.glob(pattern)
    if matches:
        print(f"⚠️  Found {len(matches)} leftover tmp file(s) — cache may be corrupt")
        for m in matches:
            print(f"     {os.path.basename(m)}")
    return len(matches) > 0


# =============================================================================
# CACHE CLEANUP
# =============================================================================

def clean_cache(directory: str) -> None:
    """
    Delete all files in the cache directory.

    Called when validation fails to force a clean cold rebuild.

    Args:
        directory: Path to the cache directory
    """
    if not os.path.isdir(directory):
        return

    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)

    print(f"🗑️  Cache cleaned: {directory}/")


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _tmp_path_for(final_path: str) -> str:
    """
    Generate a scoped temporary file path for atomic writes.

    Example: /cache/faiss.index → /cache/.finsight_tmp_faiss.index
    """
    directory = os.path.dirname(final_path)
    filename = os.path.basename(final_path)
    return os.path.join(directory, f"{TMP_PREFIX}{filename}")
