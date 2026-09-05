"""Quick test: metadata mapping (Test 3) + Test 2 info"""
import json, os, urllib.request

BASE_URL = "http://localhost:8000"
QUERY = "What are the risk factors?"

# ---- Test 3: Metadata Mapping ----
print("=" * 60)
print("TEST 3: METADATA MAPPING TEST")
print("=" * 60)

data = json.dumps({"query": QUERY, "top_k": 5}).encode()
req = urllib.request.Request(f"{BASE_URL}/retrieve", data=data, headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req, timeout=60)
results = json.loads(resp.read())["results"]

import fitz
doc = fitz.open("data/sample.pdf")
full_text = ""
for page in doc:
    full_text += page.get_text()
num_pages = doc.page_count
doc.close()
print(f"PDF: {num_pages} pages, {len(full_text):,} chars extracted\n")

all_ok = True
for i, r in enumerate(results):
    snippet = r["snippet"]
    s1 = snippet[:80].strip()
    mid = len(snippet) // 4
    s2 = snippet[mid:mid+60].strip()
    found = s1 in full_text or s2 in full_text
    all_ok = all_ok and found
    status = "FOUND" if found else "NOT FOUND"
    print(f"  {i+1}. {r['chunk_id']} - {status}")
    print(f"     snippet[:60] = \"{snippet[:60]}...\"")
    if not found:
        for start in range(0, min(len(snippet), 120), 10):
            for length in range(50, 15, -5):
                sub = snippet[start:start+length]
                if sub in full_text:
                    print(f"     partial match ({length} chars) at offset {start}")
                    break

print(f"\nTEST 3: {'PASSED' if all_ok else 'FAILED'}")

# ---- Test 2: Check for ingest endpoint ----
print("\n" + "=" * 60)
print("TEST 2: CONTAMINATION TEST (checking endpoints)")
print("=" * 60)
try:
    req2 = urllib.request.Request(f"{BASE_URL}/openapi.json")
    resp2 = urllib.request.urlopen(req2, timeout=10)
    openapi = json.loads(resp2.read())
    paths = list(openapi.get("paths", {}).keys())
    print(f"Available endpoints: {paths}")
    upload_eps = [p for p in paths if "ingest" in p.lower() or "upload" in p.lower() or "document" in p.lower()]
    if upload_eps:
        print(f"Upload endpoints found: {upload_eps}")
    else:
        print("No upload/ingest endpoint found - Test 2 requires manual setup")
except Exception as e:
    print(f"Error checking endpoints: {e}")
