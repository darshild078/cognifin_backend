"""
FinSight AI - Stage 1 Addendum: Persistence Safety Verification
=================================================================
Validates that the index cache is crash-safe, version-safe, 
order-safe, and upgrade-safe.

Usage:
    1. Start the server:  cd backend && python main.py
    2. Wait for "Server ready" message
    3. In another terminal:  cd backend && python test_persistence.py

Tests:
    1. Cache files exist in index_cache/
    2. Manifest is valid JSON with expected fields
    3. Document registry is valid with ingestion order
    4. /retrieve endpoint returns results from cached index
    5. No leftover .finsight_tmp_* files (atomic write safety)
    6. Config fingerprint and cache_format_version present
    7. Chunk order sentinels present and per-document keyed
"""

import os
import json
import glob
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"
CACHE_DIR = os.getenv("INDEX_CACHE_DIR", "index_cache")
QUERY = "What are the risk factors?"
TMP_PREFIX = ".finsight_tmp_"


def test_1_cache_files_exist():
    """Test 1: Verify cache directory and files exist."""
    print("=" * 60)
    print("TEST 1: CACHE FILES EXIST")
    print("=" * 60)

    required_files = [
        "faiss.index",
        "chunks.pkl",
        "index_manifest.json",
        "document_registry.json",
        "chunk_metadata.pkl",
    ]

    all_exist = True
    for fname in required_files:
        path = os.path.join(CACHE_DIR, fname)
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        status = f"✅ {size:,} bytes" if exists else "❌ MISSING"
        print(f"  {fname:30s} {status}")
        if not exists:
            all_exist = False

    result = "PASSED" if all_exist else "FAILED"
    print(f"\nTEST 1: {result}\n")
    return all_exist


def test_2_manifest_valid():
    """Test 2: Verify manifest has all required fields."""
    print("=" * 60)
    print("TEST 2: MANIFEST VALIDATION")
    print("=" * 60)

    manifest_path = os.path.join(CACHE_DIR, "index_manifest.json")
    if not os.path.exists(manifest_path):
        print("  ❌ Manifest file not found")
        print("\nTEST 2: FAILED\n")
        return False

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    required_keys = [
        "cache_format_version",
        "config_fingerprint",
        "source_pdf_hash",
        "embedding_model",
        "chunk_size",
        "chunk_overlap",
        "num_chunks",
        "embedding_dim",
        "chunk_order_sentinel",
        "created_at",
        "code_version",
    ]

    all_valid = True
    for key in required_keys:
        value = manifest.get(key)
        if value is None:
            print(f"  ❌ Missing key: {key}")
            all_valid = False
        else:
            display = str(value)[:60]
            print(f"  ✅ {key:25s} = {display}")

    result = "PASSED" if all_valid else "FAILED"
    print(f"\nTEST 2: {result}\n")
    return all_valid


def test_3_registry_valid():
    """Test 3: Verify document registry with ingestion order."""
    print("=" * 60)
    print("TEST 3: DOCUMENT REGISTRY + INGESTION ORDER")
    print("=" * 60)

    registry_path = os.path.join(CACHE_DIR, "document_registry.json")
    if not os.path.exists(registry_path):
        print("  ❌ Registry file not found")
        print("\nTEST 3: FAILED\n")
        return False

    with open(registry_path, "r") as f:
        registry = json.load(f)

    docs = registry.get("documents", {})
    total = registry.get("total_vectors", 0)
    order = registry.get("document_ingestion_order", [])

    print(f"  Documents registered: {len(docs)}")
    print(f"  Total vectors:        {total}")
    print(f"  Ingestion order:      {order}")

    valid = len(docs) > 0 and total > 0

    # Validate ingestion order matches documents
    if set(order) != set(docs.keys()):
        print("  ❌ Ingestion order doesn't match document keys")
        valid = False
    else:
        print("  ✅ Ingestion order matches document keys")

    for doc_id, data in docs.items():
        print(f"  Document: {doc_id}")
        print(f"    chunks: {data.get('chunk_count', '?')}")
        print(f"    vectors: {data.get('vector_id_start', '?')}-"
              f"{data.get('vector_id_end', '?')}")

    result = "PASSED" if valid else "FAILED"
    print(f"\nTEST 3: {result}\n")
    return valid


def test_4_retrieval_works():
    """Test 4: Verify /retrieve returns results."""
    print("=" * 60)
    print("TEST 4: RETRIEVAL FROM CACHED INDEX")
    print("=" * 60)

    try:
        data = json.dumps({"query": QUERY, "top_k": 3}).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/retrieve",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())

        results = result.get("results", [])
        print(f"  Query: \"{QUERY}\"")
        print(f"  Results returned: {len(results)}")

        for i, r in enumerate(results):
            print(f"    [{i+1}] {r['chunk_id']} — score={r['score']:.4f}")

        valid = len(results) > 0
        result_str = "PASSED" if valid else "FAILED"
        print(f"\nTEST 4: {result_str}\n")
        return valid

    except urllib.error.URLError as e:
        print(f"  ❌ Server not reachable: {e}")
        print("  Make sure the server is running on http://localhost:8000")
        print("\nTEST 4: FAILED\n")
        return False


def test_5_no_leftover_tmp():
    """Test 5: No .finsight_tmp_* files after successful save."""
    print("=" * 60)
    print("TEST 5: ATOMIC WRITE SAFETY")
    print("=" * 60)

    if not os.path.isdir(CACHE_DIR):
        print("  ❌ Cache directory doesn't exist")
        print("\nTEST 5: FAILED\n")
        return False

    pattern = os.path.join(CACHE_DIR, f"{TMP_PREFIX}*")
    matches = glob.glob(pattern)

    if matches:
        print(f"  ❌ Found {len(matches)} leftover tmp file(s):")
        for m in matches:
            print(f"     {os.path.basename(m)}")
        print("\nTEST 5: FAILED\n")
        return False

    print("  ✅ No leftover .finsight_tmp_* files found")
    print("\nTEST 5: PASSED\n")
    return True


def test_6_fingerprint_and_version():
    """Test 6: Config fingerprint and cache_format_version valid."""
    print("=" * 60)
    print("TEST 6: FINGERPRINT & FORMAT VERSION")
    print("=" * 60)

    manifest_path = os.path.join(CACHE_DIR, "index_manifest.json")
    if not os.path.exists(manifest_path):
        print("  ❌ Manifest not found")
        print("\nTEST 6: FAILED\n")
        return False

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    valid = True

    # Check format version is an integer
    fmt_ver = manifest.get("cache_format_version")
    if isinstance(fmt_ver, int) and fmt_ver >= 1:
        print(f"  ✅ cache_format_version = {fmt_ver}")
    else:
        print(f"  ❌ cache_format_version invalid: {fmt_ver}")
        valid = False

    # Check fingerprint is a 64-char hex string (SHA-256)
    fp = manifest.get("config_fingerprint", "")
    if len(fp) == 64 and all(c in "0123456789abcdef" for c in fp):
        print(f"  ✅ config_fingerprint = {fp[:32]}...")
    else:
        print(f"  ❌ config_fingerprint invalid: {fp}")
        valid = False

    result = "PASSED" if valid else "FAILED"
    print(f"\nTEST 6: {result}\n")
    return valid


def test_7_chunk_sentinels():
    """Test 7: Per-document chunk order sentinels present."""
    print("=" * 60)
    print("TEST 7: CHUNK ORDER SENTINELS")
    print("=" * 60)

    manifest_path = os.path.join(CACHE_DIR, "index_manifest.json")
    if not os.path.exists(manifest_path):
        print("  ❌ Manifest not found")
        print("\nTEST 7: FAILED\n")
        return False

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    sentinels = manifest.get("chunk_order_sentinel")
    if not sentinels or not isinstance(sentinels, dict):
        print("  ❌ chunk_order_sentinel missing or invalid")
        print("\nTEST 7: FAILED\n")
        return False

    valid = True
    for doc_id, data in sentinels.items():
        first = data.get("first", [])
        last = data.get("last", [])
        print(f"  Document: {doc_id}")
        print(f"    first hashes: {len(first)} "
              f"({first[0][:16]}... )" if first else "    first: EMPTY")
        print(f"    last hashes:  {len(last)} "
              f"({last[0][:16]}... )" if last else "    last: EMPTY")

        if not first or not last:
            print(f"  ❌ Missing sentinel hashes for {doc_id}")
            valid = False
        else:
            # Verify they're 40-char SHA-1 hex strings
            all_valid_hashes = all(
                len(h) == 40 and all(c in "0123456789abcdef" for c in h)
                for h in first + last
            )
            if all_valid_hashes:
                print(f"    ✅ All hashes are valid SHA-1")
            else:
                print(f"    ❌ Some hashes are not valid SHA-1")
                valid = False

    result = "PASSED" if valid else "FAILED"
    print(f"\nTEST 7: {result}\n")
    return valid


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FinSight AI — Persistence Safety Verification")
    print("=" * 60 + "\n")

    results = {
        "Cache Files":       test_1_cache_files_exist(),
        "Manifest":          test_2_manifest_valid(),
        "Registry + Order":  test_3_registry_valid(),
        "Retrieval":         test_4_retrieval_works(),
        "Atomic Writes":     test_5_no_leftover_tmp(),
        "Fingerprint":       test_6_fingerprint_and_version(),
        "Sentinels":         test_7_chunk_sentinels(),
    }

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:20s} {status}")
        if not passed:
            all_passed = False

    print(f"\nOverall: {'ALL PASSED ✅' if all_passed else 'SOME FAILED ❌'}\n")
