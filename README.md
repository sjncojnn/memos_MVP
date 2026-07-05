# MemOS-lite MVP — Memory Management for Business QA

MemOS-lite is a local MVP for question answering over business documents. It extends a normal RAG pipeline with a lightweight memory management layer, so knowledge is not only retrieved, but also stored, tracked, updated, expired, prioritized, and reused over time.

The project is designed for stage 1 of the problem statement: business document QA for ViettelPay Pro. It runs locally on macOS without GPU using Ollama, SQLite, Python, and Streamlit.

---

## 1. Main Idea

A normal RAG system stores documents as flat chunks. It can retrieve relevant chunks, but it usually does not manage knowledge lifecycle, duplication, conflict, usage frequency, or repeated questions.

MemOS-lite treats each chunk or FAQ item as a **memory unit**. A memory unit contains:

- content and embedding
- source file and source reference
- category and subcategory
- lifecycle status: `active`, `expired`, `superseded`
- retrieval tier: `hot`, `warm`, `cold`
- access count and last access time
- optional TTL
- content hash for exact duplicate detection

This makes the system closer to a managed memory layer rather than a simple document retrieval script.

---

## 2. What This MVP Supports

| Area | Implementation |
|---|---|
| Document ingestion | Reads `.docx` business documents and `.xlsx` FAQ files |
| Text processing | Cleans text, chunks documents, assigns categories, stores provenance |
| Memory API | Uses `MemoryStore` as the unified API for add/search/update/delete |
| Duplicate handling | Skips exact duplicates using content hash |
| Conflict handling | Detects near-duplicate candidates and stores them in a conflict queue |
| Lifecycle management | Supports expired and superseded memory units |
| Hot/cold tiering | Promotes frequently used units to hot tier and moves old/rarely used units to cold tier |
| QA service | Retrieves memory units, calls LLM, and returns answers with sources |
| Semantic QA cache | Reuses stable answers for repeated or similar question intents |
| Baseline RAG | Provides a conventional RAG baseline for comparison |
| Evaluation | Measures EM, F1, LLM-judge score, latency, and cache-hit rate |
| Demo UI | Provides a Streamlit interface for ingestion, QA, batch testing, monitoring, and conflict review |

---

## 3. Repository Structure

```text
.
├── config.py              # Central configuration
├── db.py                  # SQLite schema for MemOS-lite
├── vector_store.py        # Embedding encode/decode and cosine similarity
├── ollama_client.py       # Ollama backend for chat, embedding, and judge
├── llamacpp_client.py     # Optional llama.cpp backend
├── client_factory.py      # Selects backend: ollama or llamacpp
├── ingest.py              # Parse .docx/.xlsx, clean text, chunk, assign category
├── memory_api.py          # MemoryStore: unified memory API and conflict workflow
├── qa_service.py          # QA pipeline with retrieval and semantic QA cache
├── scheduler.py           # Hot/warm/cold tiering and TTL expiration
├── cli.py                 # CLI for MemOS-lite
├── app_streamlit.py       # Streamlit demo app
├── eval.py                # Evaluation for one system
├── eval_compare.py        # Compare MemOS-lite with RAG baseline
├── baseline_db.py         # SQLite schema for baseline RAG
├── baseline_store.py      # Simple baseline store without memory management
├── baseline_ingest.py     # Baseline ingestion using the same parser
├── baseline_qa.py         # Baseline QA: retrieval + LLM every time
├── baseline_cli.py        # CLI for baseline RAG
└── requirements.txt
```

---

## 4. Installation

### 4.1. Create Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4.2. Install and run Ollama

```bash
brew install ollama
ollama serve
```

If Ollama is already running as an app, you do not need to run `ollama serve` again.

Pull the default models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Default model settings are defined in `config.py`:

```python
CHAT_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
```

You can override them with environment variables:

```bash
export MEMOS_CHAT_MODEL=llama3.2:3b
export MEMOS_EMBED_MODEL=nomic-embed-text
```

---

## 5. Prepare Data

Recommended data layout:

```text
data_raw/
├── tai_lieu_nghiep_vu/
│   ├── file_1.docx
│   ├── file_2.docx
│   └── ...
├── faq.xlsx
└── golden_set.csv
```

The golden set for evaluation should contain:

```csv
question,reference_answer,topic
```

`topic` is optional.

---

## 6. Run MemOS-lite from CLI

Initialize the database:

```bash
python cli.py init-db
```

Ingest business documents:

```bash
python cli.py ingest-docs ./data_raw/tai_lieu_nghiep_vu
```

Ingest FAQ:

```bash
python cli.py ingest-faq ./data_raw/faq.xlsx
```

Ask one question:

```bash
python cli.py ask "Điều kiện vay tiền qua ViettelPay Pro là gì?"
```

Check memory statistics:

```bash
python cli.py stats
```

Run lifecycle scheduler:

```bash
python cli.py tier
```

Preview scheduler changes without writing to the database:

```bash
python cli.py tier --dry-run
```

---

## 7. Run the Streamlit Demo

```bash
streamlit run app_streamlit.py
```

The demo UI includes:

1. **Upload & Ingest**: upload `.docx` and `.xlsx` files.
2. **Single QA Demo**: ask one question and view answer, sources, cache status, and latency.
3. **Batch Questions**: upload a batch of questions and export results.
4. **Memory Monitor**: view memory statistics, QA cache, ingest logs, and tiering status.
5. **Duplicate / Conflict Monitor**: review near-duplicate conflict candidates.

The sidebar also supports demo mode, which lowers some thresholds so cache, tiering, and conflict behavior can be observed faster.

---

## 8. Evaluation

### 8.1. Evaluate MemOS-lite only

```bash
python cli.py eval ./data_raw/golden_set.csv
```

Without LLM judge:

```bash
python cli.py eval ./data_raw/golden_set.csv --no-judge
```

Output:

- detailed CSV file, default: `eval_results.csv`
- console summary with:
  - number of questions
  - EM
  - F1
  - average LLM-judge score
  - average cache-miss latency
  - average cache-hit latency
  - cache-hit rate

### 8.2. Compare with RAG baseline

The baseline uses the same parser, embedding method, and LLM backend, but does not include memory management features.

Initialize and ingest baseline data:

```bash
python baseline_cli.py init-db
python baseline_cli.py ingest-docs ./data_raw/tai_lieu_nghiep_vu
python baseline_cli.py ingest-faq ./data_raw/faq.xlsx
```

Run comparison:

```bash
python eval_compare.py ./data_raw/golden_set.csv
```

Without LLM judge:

```bash
python eval_compare.py ./data_raw/golden_set.csv --no-judge
```

The comparison reports:

| Metric | Meaning |
|---|---|
| EM | Exact match against reference answer |
| F1 | Token-level overlap against reference answer |
| LLM_judge_avg_1_5 | Local LLM score from 1 to 5 |
| avg_latency_cache_miss_sec | Average latency when cache is not used |
| avg_latency_cache_hit_sec | Average latency when cache is used |
| cache_hit_rate | Ratio of questions answered from cache |

Because the baseline has no QA cache, its cache-hit rate should remain 0.

---

## 9. Baseline vs MemOS-lite

| Feature | MemOS-lite | RAG baseline |
|---|---|---|
| Same document parser | Yes | Yes |
| Same embedding backend | Yes | Yes |
| Same LLM backend | Yes | Yes |
| Add/search knowledge | Yes | Yes |
| Update/delete knowledge | Yes | No |
| Exact duplicate detection | Yes | No |
| Near-duplicate conflict queue | Yes | No |
| Lifecycle status | Yes | No |
| Hot/warm/cold tiering | Yes | No |
| TTL expiration | Yes | No |
| Semantic QA cache | Yes | No |
| Source reference | Yes | Yes |

This setup makes the comparison focus mainly on the value of the memory management layer.

---

## 10. Optional: llama.cpp Backend

Ollama is the default backend. The project also includes `llamacpp_client.py` for optional llama.cpp server usage.

Set backend:

```bash
export MEMOS_LLM_BACKEND=llamacpp
export MEMOS_LLAMACPP_HOST=http://localhost:8080
```

Run with CLI:

```bash
python cli.py --backend llamacpp ask "Điều kiện vay tiền qua ViettelPay Pro là gì?"
```

Note: semantic QA cache works in both backends. Direct low-level KV-cache control is not available through Ollama. llama.cpp can demonstrate prefix/prompt cache behavior depending on server configuration and version.

---

## 11. Configuration

Important settings are in `config.py`:

| Setting | Purpose |
|---|---|
| `CHAT_MODEL` | Ollama chat model |
| `EMBED_MODEL` | Ollama embedding model |
| `TOP_K` | Number of retrieved memory units |
| `SIM_THRESHOLD_MIN` | Minimum retrieval similarity |
| `NEAR_DUP_THRESHOLD` | Threshold for near-duplicate candidates |
| `HOT_ACCESS_THRESHOLD` | Access count needed for hot tier |
| `COLD_AFTER_DAYS` | Days without access before cold tier |
| `QA_CACHE_MATCH_THRESHOLD` | Similarity threshold for same question intent |
| `STABLE_CACHE_MIN_HITS` | Hits needed before a cached answer is stable |
| `QA_CACHE_MAX_ITEMS` | Maximum number of QA cache entries |
| `QA_CACHE_TTL_DAYS` | Time-to-live for QA cache entries |

---

## 12. Known Limitations

This MVP is intentionally limited to stage 1.

It does not include:

- fine-grained access control
- full governance and audit workflow
- multi-agent memory sharing
- cross-platform memory migration
- LoRA-based or parameter-level memory internalization
- true manual control of raw KV-cache in Ollama
- production-scale vector database deployment

Important implementation limits:

- Near-duplicate detection is only a warning signal, not automatic business conflict resolution.
- Hot/cold tiering uses simple rules based on access count and time, not machine learning.
- Semantic QA cache can reduce latency, but cache safety depends on similarity threshold and hit count.
- SQLite + numpy cosine search is suitable for small stage-1 data, but should be replaced by a stronger vector backend for larger datasets.

---

## 13. Quick Troubleshooting

### Ollama connection refused

Make sure Ollama is running:

```bash
ollama serve
```

Then check:

```bash
curl http://localhost:11434/api/tags
```

### Missing model

Pull the required models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Evaluation is slow

Use `--no-judge` to skip LLM-judge:

```bash
python eval_compare.py ./data_raw/golden_set.csv --no-judge
```

### Cache does not hit

Check:

- `QA_CACHE_MATCH_THRESHOLD`
- `STABLE_CACHE_MIN_HITS`
- whether the questions are semantically similar enough
- whether the same database is being used

---

## 14. Development Check

The current Python files pass syntax compilation:

```bash
python -m py_compile *.py
```

For real QA quality, run the system with Ollama and evaluate on the golden question-answer set.

---

## 15. Project Status

This repository provides a working local MVP, not a production-ready MemOS system. The main goal is to demonstrate that a conventional RAG pipeline can be extended with a lightweight memory management layer for long-term business QA.
