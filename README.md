# MemOS-lite — MVP Quản lý Bộ nhớ cho LLM (giai đoạn 1)

MVP rút gọn từ kiến trúc [MemOS](https://github.com/MemTensor/MemOS) (MemCube / Unified
Memory API / MemScheduler), chạy **hoàn toàn local trên macOS, không cần GPU**:
**Ollama** (LLM + embedding) + **SQLite** (metadata + vector, dùng numpy cosine similarity
thay Neo4j/Qdrant) + **Python thuần**, không Docker / Redis / cloud / multimodal /
multi-agent / LoRA / phân quyền phức tạp.

Đáp ứng đầy đủ **giai đoạn 1** của Problem Statement (`Problem_Statement_Memory_Management_LLM.docx`):

| Yêu cầu mục 4 | Hiện thực trong MVP |
|---|---|
| Tiếp nhận & chuẩn hoá tri thức | `ingest.py`: đọc .docx (theo heading) + FAQ .xlsx (tự dò cột), làm sạch mojibake, chunk, gắn category |
| Truy hồi & giao diện thống nhất | `memory_api.py`: `MemoryStore.add/search/update/delete/get_by_id` — điểm chạm DUY NHẤT tới dữ liệu |
| Lưu trữ dài hạn xuyên phiên, vòng đời rút gọn | SQLite bền vững giữa các lần chạy; `status`: active/cold/expired; `tier`: hot/warm/cold |
| Tối ưu hiệu năng (tái sử dụng dạng trạng thái suy luận) | `qa_service.py`: cache câu trả lời ổn định (semantic cache, bỏ qua gọi LLM khi trúng) + system prompt cố định để Ollama tái dùng KV-cache nội bộ cho phần prefix chung |
| Ghi vết nguồn gốc tối thiểu | mỗi knowledge unit lưu `source_file`, `source_ref`, `created_at`, `content_hash`, `version` |
| Tự động hoá phân tầng nóng/lạnh | `scheduler.py`: `run_tiering()` — luật đơn giản dựa trên `access_count` + thời gian truy cập gần nhất |

Đánh giá (mục 5) qua `eval.py`: EM/F1 so với đáp án tham chiếu, LLM-judge (dùng chính model
local để chấm 1–5), độ trễ cache-hit vs cache-miss, xuất `eval_results.csv`.

---

## 1. Kiến trúc thư mục

```
memos-mvp/
├── config.py         # toàn bộ tham số cấu hình (model, ngưỡng dedup, TTL, hot/cold...)
├── db.py             # schema SQLite (knowledge_units, qa_cache, ingest_log)
├── vector_store.py   # cosine similarity trên numpy, embedding lưu trong SQLite
├── ollama_client.py  # gọi Ollama REST API (embeddings + chat + judge)
├── ingest.py         # parse .docx / .xlsx, làm sạch, chunk, gắn nhãn
├── memory_api.py      # MemoryStore: add / search / update / delete (unified API)
├── scheduler.py       # phân tầng nóng/lạnh + hết hạn TTL
├── qa_service.py       # pipeline RAG + cache câu trả lời ổn định
├── eval.py             # EM/F1 + LLM-judge + báo cáo
├── cli.py               # entrypoint duy nhất (xem mục 4)
├── requirements.txt
└── tests/
    ├── fake_ollama.py       # client giả lập để test KHÔNG cần Ollama server
    ├── make_sample_data.py  # sinh dữ liệu mẫu nhỏ
    └── test_pipeline.py     # 30 test end-to-end (đã chạy PASS 30/30, xem mục 5)
```

## 2. Cài đặt trên macOS

```bash
# 1) Cài Ollama (nếu chưa có)
brew install ollama
ollama serve &                     # hoặc mở app Ollama

# 2) Tải model (chọn model phù hợp cấu hình máy, càng nhỏ chạy CPU càng nhanh)
ollama pull qwen2.5:7b             # model chat — có thể đổi sang qwen2.5:3b/phi3.5 nếu máy yếu
ollama pull nomic-embed-text       # model embedding

# 3) Cài Python deps
cd memos-mvp
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Nếu đổi model, sửa `config.py` (`CHAT_MODEL`, `EMBED_MODEL`, `EMBED_DIM`) hoặc set biến môi
trường `MEMOS_CHAT_MODEL` / `MEMOS_EMBED_MODEL`.

## 3. Chuẩn bị dữ liệu

Đặt 16 file `.docx` nghiệp vụ vào 1 thư mục (vd `data_raw/tai_lieu_nghiep_vu/`) và file FAQ
`.xlsx` (vd `data_raw/faq.xlsx`) với cột chứa "câu hỏi" và "trả lời" (hệ thống tự dò tên cột,
không phân biệt hoa/thường/dấu). `ingest.py` tự đoán category theo tên file qua từ khoá trong
`config.guess_category()` — có thể chỉnh sửa rule cho khớp tên 16 file thật.

## 4. Chạy

```bash
python cli.py init-db
python cli.py ingest-docs ./data_raw/tai_lieu_nghiep_vu
python cli.py ingest-faq ./data_raw/faq.xlsx
python cli.py stats
python cli.py ask "Điều kiện vay tiền qua ViettelPay Pro là gì?"
python cli.py tier              # chạy phân tầng nóng/lạnh + TTL (nên đặt cron/launchd định kỳ)
python cli.py eval ./data_raw/golden_set.csv       # bộ đánh giá vàng: question,reference_answer,topic
```

## 5. Đã tự kiểm tra (recheck) như thế nào

Vì sandbox này **không có internet và không chạy được Ollama server thật**, mình đã:

1. Viết `tests/fake_ollama.py` — client giả lập embedding (bag-of-words hash, xác định) +
   chat + judge, để kiểm chứng **toàn bộ logic hệ thống** (không kiểm chứng chất lượng
   ngôn ngữ tự nhiên — phần đó cần Ollama thật trên máy bạn).
2. Sinh dữ liệu mẫu nhỏ (`tests/make_sample_data.py`): 3 file .docx (gồm 1 file gần trùng
   nội dung để test near-dup), 1 file FAQ .xlsx, 1 golden set.
3. Chạy `python tests/test_pipeline.py` → **30/30 test PASS**, bao gồm:
   - ingest .docx/.xlsx đúng số lượng, không lỗi
   - exact-dedup (ingest lại toàn bộ bị skip)
   - near-dup detection (phát hiện + đánh version/supersede)
   - unified search trả đúng category liên quan
   - update/delete (soft-delete) hoạt động đúng, version tăng
   - TTL expiry qua scheduler
   - hot/cold tiering theo access_count
   - QA cache: hỏi lặp lại nhiều lần → chuyển sang cache_hit, latency giảm
   - eval EM/F1 + LLM-judge chạy hết bộ golden set, xuất CSV
   - làm sạch mojibake / dòng trống thừa
4. `python -m py_compile` toàn bộ module — không lỗi cú pháp.
5. Chạy thử `python cli.py --help` và `python cli.py init-db` thật (không cần Ollama) — OK.

**Việc bạn cần tự chạy thêm trên máy có Ollama**: `ask`, `ingest-docs/ingest-faq` với dữ liệu
thật, và `eval` với golden set thật — vì các lệnh này gọi Ollama thật để lấy embedding/sinh
câu trả lời, không thể chạy trong sandbox này. Nếu gặp lỗi khi đó (vd sai tên model, Ollama
chưa `serve`), thông báo lỗi từ `ollama_client.py` sẽ in rõ status code + nội dung response
để debug.

## 6. Giới hạn đã biết / phạm vi ngoài giai đoạn 1

- Hot/cold dựa trên luật đơn giản (access_count + thời gian), không dùng ML — đúng tinh thần
  "đơn giản" nhưng kém chính xác hơn nếu truy vấn dồn dập trong thời gian ngắn thay vì rải đều.
- "Tái sử dụng trạng thái suy luận (KV)" được hiện thực bằng cache câu trả lời ngữ nghĩa +
  system prompt cố định để tận dụng cơ chế cache prefix có sẵn của Ollama/llama.cpp — **không**
  thao tác trực tiếp KV-cache thô (Ollama không expose API public cho việc này). Nếu muốn true
  KV-cache reuse, cần chuyển sang chạy model qua `llama.cpp` server với `--prompt-cache` hoặc
  vLLM prefix caching.
- Near-dup dedup so sánh trong cùng category, ngưỡng cosine cố định (`NEAR_DUP_THRESHOLD`) —
  chưa có quy trình xác nhận thủ công xung đột (nằm ở "Governance đầy đủ", ngoài phạm vi mục 4).
- Không có phân quyền, đa tenant, đa mô hình, multimodal (đúng yêu cầu đã loại khỏi MVP).
