# 📊 Dataset & Index Generation Guide

CogniFin AI includes a comprehensive pre-indexed vector corpus of Indian public company filings.

---

## 📈 Dataset Overview

The corpus covers **139 comprehensive Indian financial documents** representing **47 major enterprises** across sectors (Automotive, IT, Energy, Financials, Steel, FMCG, Pharmaceuticals):

- **Sources:** Official SEBI filings, BSE/NSE annual reports, Draft Red Herring Prospectuses (DRHP), and Red Herring Prospectuses (RHP).
- **Time Horizon:** FY2020 through FY2025.
- **Scale:** 
  - **139 Document Entities**
  - **233,706 Text Chunks**
  - **233,706 768-dimensional Vector Embeddings**
  - **Total Index Cache Size:** ~1.2 GB

### Sample Companies in Corpus
- **Automotive:** Tata Motors, Mahindra & Mahindra, Maruti Suzuki, Bajaj Auto
- **Technology:** Tata Consultancy Services (TCS), Infosys, Wipro, HCL Tech
- **Energy & Conglomerates:** Reliance Industries, Adani Enterprises, Adani Green, NTPC
- **Banking & Finance:** HDFC Bank, ICICI Bank, State Bank of India (SBI)
- **Metals & Mining:** Tata Steel, JSW Steel, Hindalco

---

## 📦 What's Inside `index_cache/`

The `backend/index_cache/` directory contains all pre-computed artifacts required for zero-latency startup:

| File | Size | Description |
|---|---|---|
| **`faiss.index`** | `718 MB` | Binary FAISS index storing 233,706 768-d vectors. |
| **`chunks.pkl`** | `247 MB` | Serialized text content of every document chunk. |
| **`chunk_metadata.pkl`** | `22.8 MB` | Structured metadata (`company`, `year`, `doc_type`, `page_number`, `pdf_url`). |
| **`bm25.pkl`** | `244 MB` | Pre-computed BM25 token frequencies for instant sparse search boot. |
| **`document_registry.json`** | `51 KB` | Document catalog tracking file hashes, page counts, and metadata. |
| **`lookup_index.json`** | `28 KB` | Inverted index mapping company names and years to document IDs. |

---

## ⚙️ How Chunking Works

Financial documents have dense tables, footnotes, and multi-column layouts. The chunking pipeline in CogniFin AI uses **Layout-Aware Semantic Chunking**:

1. **PDF Extraction:** `PyMuPDF` (`fitz`) extracts text blocks while preserving reading order and table structure.
2. **Chunk Size:** `1,000 characters` (preserves table context and financial statements).
3. **Overlap:** `200 characters` (prevents boundary loss across sentences and numeric data).
4. **Metadata Preservation:** Every chunk retains its exact `page_number`, `company_name`, `financial_year`, and `document_type`.

---

## 🔄 Rebuilding or Adding New Documents

To ingest a new document (e.g., `data/sample.pdf`):

```bash
# Ingest a single PDF file
python ingestion/ingest.py --file data/sample.pdf --company "Tata Motors" --year 2024 --type "Annual Report"
```

To run batch ingestion across an entire folder:

```bash
python ingestion/batch_ingest_annual_reports.py --dir data/annual_reports/
```

The script will automatically:
1. Extract text and compute layout-aware chunks.
2. Generate 768-d embeddings using `BAAI/bge-base-en-v1.5`.
3. Update `faiss.index`, `chunks.pkl`, `chunk_metadata.pkl`, and rebuild `bm25.pkl`.
