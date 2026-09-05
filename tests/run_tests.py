"""
FinSightAI Retrieval Pipeline Tests
====================================
Tests:
1. Deterministic Retrieval Test - checks if results are stable across server restarts
2. Second Document Contamination Test - checks corpus isolation
3. Metadata Mapping Test - checks snippet accuracy against PDF

Usage: python run_tests.py
Requires: server running on http://localhost:8000
"""

import json
import time
import sys
import os
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"
QUERY = "What are the risk factors?"
TOP_K = 5


def api_call(endpoint, method="GET", body=None):
    """Make an API call and return parsed JSON."""
    url = f"{BASE_URL}{endpoint}"
    if body:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url)
    
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        return {"error": e.code, "detail": body_text}
    except urllib.error.URLError as e:
        return {"error": "connection", "detail": str(e.reason)}


def wait_for_server(max_wait=120):
    """Wait until the server is healthy and has indexed chunks."""
    print("   Waiting for server to be healthy and indexed...", end="", flush=True)
    start = time.time()
    while time.time() - start < max_wait:
        try:
            health = api_call("/health")
            if health.get("indexed") and health.get("num_chunks", 0) > 0:
                print(f" OK ({health['num_chunks']} chunks, {time.time()-start:.1f}s)")
                return True
        except:
            pass
        time.sleep(2)
        print(".", end="", flush=True)
    print(" TIMEOUT!")
    return False


def test_1_deterministic_retrieval():
    """Test 1: Deterministic Retrieval Test"""
    print("\n" + "=" * 70)
    print("TEST 1: DETERMINISTIC RETRIEVAL TEST")
    print("=" * 70)
    print(f"Query: \"{QUERY}\"")
    print()

    # --- Run 1 ---
    print("📋 RUN 1: Querying /retrieve ...")
    run1 = api_call("/retrieve", method="POST", body={"query": QUERY, "top_k": TOP_K})
    if "error" in run1:
        print(f"   ❌ ERROR: {run1}")
        return False

    results1 = run1["results"]
    print(f"   Got {len(results1)} results:")
    for i, r in enumerate(results1):
        print(f"   {i+1}. {r['chunk_id']}  score={r['score']}")

    # --- Trigger server restart via file touch ---
    print("\n🔄 Triggering server restart (touching main.py for --reload)...")
    # Touch the file to trigger uvicorn --reload
    main_path = os.path.join(os.path.dirname(__file__), "main.py")
    os.utime(main_path, None)
    
    # Wait for the server to go down and come back up
    time.sleep(5)  # Give uvicorn time to detect the change
    if not wait_for_server(120):
        print("   ❌ Server did not restart in time!")
        return False

    # --- Run 2 ---
    print("\n📋 RUN 2: Querying /retrieve again after restart ...")
    run2 = api_call("/retrieve", method="POST", body={"query": QUERY, "top_k": TOP_K})
    if "error" in run2:
        print(f"   ❌ ERROR: {run2}")
        return False

    results2 = run2["results"]
    print(f"   Got {len(results2)} results:")
    for i, r in enumerate(results2):
        print(f"   {i+1}. {r['chunk_id']}  score={r['score']}")

    # --- Compare ---
    print("\n🔍 COMPARING RESULTS:")
    ids1 = [r["chunk_id"] for r in results1]
    ids2 = [r["chunk_id"] for r in results2]
    scores1 = [r["score"] for r in results1]
    scores2 = [r["score"] for r in results2]

    order_match = ids1 == ids2
    score_match = scores1 == scores2

    print(f"   Run 1 order: {ids1}")
    print(f"   Run 2 order: {ids2}")
    print(f"   Order match: {'✅ YES' if order_match else '❌ NO'}")
    print(f"   Score match: {'✅ YES' if score_match else '❌ NO (scores differ)'}")

    if not order_match:
        # Show what changed
        for i in range(max(len(ids1), len(ids2))):
            id1 = ids1[i] if i < len(ids1) else "MISSING"
            id2 = ids2[i] if i < len(ids2) else "MISSING"
            marker = "  ✅" if id1 == id2 else "  ❌ CHANGED"
            print(f"   Position {i+1}: {id1} → {id2}{marker}")

    passed = order_match
    print(f"\n{'✅ TEST 1 PASSED' if passed else '❌ TEST 1 FAILED'}: "
          f"{'Results are deterministic across restarts' if passed else 'Results differ after restart — nondeterministic index!'}")
    return passed


def test_3_metadata_mapping():
    """Test 3: Metadata Mapping Test — verify snippets exist in original PDF."""
    print("\n" + "=" * 70)
    print("TEST 3: METADATA MAPPING TEST")
    print("=" * 70)

    # Get retrieval results
    print("📋 Querying /retrieve ...")
    resp = api_call("/retrieve", method="POST", body={"query": QUERY, "top_k": TOP_K})
    if "error" in resp:
        print(f"   ❌ ERROR: {resp}")
        return False

    results = resp["results"]
    
    # Try to extract text from the PDF and verify snippets
    pdf_path = os.path.join(os.path.dirname(__file__), "data", "sample.pdf")
    if not os.path.exists(pdf_path):
        print(f"   ⚠️  PDF not found at {pdf_path}, cannot verify snippets.")
        return False
    
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("   ⚠️  PyMuPDF not installed, cannot verify snippets.")
        return False
    
    print(f"📄 Loading PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    num_pages = doc.page_count
    doc.close()
    print(f"   Extracted {len(full_text):,} characters from {num_pages} pages")
    
    print("\n🔍 Verifying each snippet exists in the original PDF:\n")
    all_found = True
    for i, r in enumerate(results):
        snippet = r["snippet"]
        # Use a substring (first 80 chars, cleaned) for matching 
        # because chunk extraction may have slight formatting diffs
        search_text = snippet[:80].strip()
        
        # Also try a middle portion in case start has formatting issues
        mid_start = len(snippet) // 4
        mid_text = snippet[mid_start:mid_start + 60].strip()
        
        found_start = search_text in full_text
        found_mid = mid_text in full_text
        found = found_start or found_mid
        
        status = "✅ FOUND" if found else "❌ NOT FOUND"
        all_found = all_found and found
        
        print(f"   {i+1}. {r['chunk_id']} — {status}")
        print(f"      Snippet start: \"{search_text[:60]}...\"")
        if not found:
            print(f"      ⚠️  Could not locate this text in the PDF!")
            # Try fuzzy: find longest matching substring
            best_len = 0
            for start in range(0, min(len(snippet), 100), 10):
                for length in range(40, 10, -1):
                    substr = snippet[start:start+length]
                    if substr in full_text:
                        if length > best_len:
                            best_len = length
                            print(f"      Partial match ({length} chars) at offset {start}: \"{substr[:50]}...\"")
                        break
        print()

    print(f"{'✅ TEST 3 PASSED' if all_found else '❌ TEST 3 FAILED'}: "
          f"{'All snippets verified in original PDF' if all_found else 'Some snippets not found — vector_id mapping may be wrong!'}")
    return all_found


def main():
    print("=" * 70)
    print("  FinSightAI Retrieval Pipeline Tests")
    print("=" * 70)
    
    # Check server is up
    if not wait_for_server(30):
        print("\n❌ Server is not running! Start it with: uvicorn main:app --reload")
        sys.exit(1)
    
    results = {}
    
    # Test 1: Deterministic Retrieval
    results["test_1"] = test_1_deterministic_retrieval()
    
    # Test 3: Metadata Mapping (run before Test 2 since Test 2 adds a second doc)
    results["test_3"] = test_3_metadata_mapping()
    
    # Note: Test 2 (contamination) requires adding a second PDF
    # We'll check if there's a way to do it via the API
    print("\n" + "=" * 70)
    print("TEST 2: SECOND DOCUMENT CONTAMINATION TEST")
    print("=" * 70)
    print("⚠️  This test requires adding a second PDF to the system.")
    print("   The current server only indexes the single PDF from .env at startup.")
    print("   To run this test manually:")
    print("   1. Add a second PDF to backend/data/ (any PDF, even a textbook)")
    print("   2. Use the corpus_manager.add_document() API if available")
    print("   3. Or check if there's an /ingest endpoint")
    
    # Check if there's an ingest endpoint
    print("\n   Checking for document upload endpoints...")
    try:
        openapi = api_call("/openapi.json")
        paths = openapi.get("paths", {})
        upload_endpoints = [p for p in paths if "ingest" in p.lower() or "upload" in p.lower() or "document" in p.lower()]
        if upload_endpoints:
            print(f"   Found endpoints: {upload_endpoints}")
        else:
            print("   No upload/ingest endpoint found.")
            print("   Test 2 requires code-level changes or adding a second PDF manually.")
    except:
        print("   Could not check OpenAPI spec.")
    
    results["test_2"] = None  # Not automated
    
    # Summary
    print("\n" + "=" * 70)
    print("  TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results.items():
        if passed is None:
            status = "⏭️  SKIPPED (manual test needed)"
        elif passed:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        print(f"  {test_name}: {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
