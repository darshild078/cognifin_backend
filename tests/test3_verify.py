"""Test 3: Metadata Mapping - using correct page join method"""
import fitz, urllib.request, json

# Extract text EXACTLY as the pipeline does (join with \n\n)
doc = fitz.open("data/sample.pdf")
parts = [p.get_text() for p in doc]
num_pages = doc.page_count
doc.close()
full = "\n\n".join(parts)
print(f"PDF: {num_pages} pages, {len(full):,} chars")

# Get retrieval results
data = json.dumps({"query": "What are the risk factors?", "top_k": 5}).encode()
req = urllib.request.Request(
    "http://localhost:8000/retrieve", data=data,
    headers={"Content-Type": "application/json"}
)
resp = json.loads(urllib.request.urlopen(req).read())

print("\nSnippet verification:")
all_ok = True
for r in resp["results"]:
    s = r["snippet"]
    # Check multiple substrings
    found = (s[:80] in full) or (s[50:130] in full) or (s[100:180] in full)
    status = "FOUND" if found else "NOT FOUND"
    all_ok = all_ok and found
    print(f"  {r['chunk_id']} (score={r['score']}): {status}")
    if not found:
        # Debug: show what chars differ
        for i in range(0, min(len(s), 200), 20):
            sub = s[i:i+30]
            if sub in full:
                print(f"    partial FOUND at offset {i}: {repr(sub[:30])}")
            else:
                print(f"    NOT in PDF at offset {i}: {repr(sub[:30])}")

print(f"\nTEST 3: {'PASSED' if all_ok else 'FAILED'}")
