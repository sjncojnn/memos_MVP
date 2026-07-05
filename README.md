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
| Lưu trữ dài hạn xuyên phiên, vòng đời rút gọn | SQLite bền vững giữa các lần chạy; `status`: active/cold/expired/superseded; `tier`: hot/warm/cold |
| Tối ưu hiệu năng (tái sử dụng dạng trạng thái suy luận) | `qa_service.py`: cache câu trả lời ổn định (semantic cache, bỏ qua gọi LLM khi trúng) + system prompt cố định; backend `llamacpp` (mục 7) demo prefix-cache thật |
| Ghi vết nguồn gốc tối thiểu | mỗi knowledge unit lưu `source_file`, `source_ref`, `created_at`, `content_hash`, `version` |
| Tự động hoá phân tầng nóng/lạnh | `scheduler.py`: `run_tiering()` — luật đơn giản dựa trên `access_count` + thời gian truy cập gần nhất |
| Khử trùng lặp & xử lý xung đột | exact-dedup tự động (hash) + near-dup đưa vào **hàng đợi conflict** (bảng `conflicts`), người dùng tự quyết định qua UI/API — xem mục 6 |

Đánh giá (mục 5) qua `eval.py`: EM/F1 so với đáp án tham chiếu, LLM-judge (dùng chính model
local để chấm 1–5), độ trễ cache-hit vs cache-miss, xuất `eval_results.csv`.

Ngoài CLI, có sẵn **Streamlit demo app** (`app_streamlit.py`) — xem mục 6.

---

## 1. Kiến trúc thư mục

```
memos-mvp/
├── config.py           # toàn bộ tham số cấu hình (model, ngưỡng dedup, TTL, hot/cold, backend...)
├── db.py               # schema SQLite (knowledge_units, qa_cache, conflicts, ingest_log)
├── vector_store.py     # cosine similarity trên numpy, embedding lưu trong SQLite
├── ollama_client.py    # gọi Ollama REST API (embeddings + chat + judge)
├── llamacpp_client.py  # gọi llama.cpp server (embeddings + chat + judge), demo prefix-cache thật
├── client_factory.py   # chọn backend (ollama | llamacpp) qua config/env/CLI/sidebar
├── ingest.py            # parse .docx / .xlsx, làm sạch, chunk, gắn nhãn
├── memory_api.py         # MemoryStore: add / search / update / delete + conflict workflow (unified API)
├── scheduler.py           # phân tầng nóng/lạnh + hết hạn TTL
├── qa_service.py           # pipeline RAG + cache câu trả lời ổn định
├── eval.py                 # EM/F1 + LLM-judge + báo cáo
├── cli.py                   # entrypoint CLI (xem mục 4)
├── app_streamlit.py          # Streamlit demo app (xem mục 6)
├── requirements.txt
└── tests/
    ├── fake_ollama.py       # client giả lập để test KHÔNG cần Ollama server
    ├── make_sample_data.py  # sinh dữ liệu mẫu nhỏ
    └── test_pipeline.py     # test end-to-end
```

## 2. Cài đặt trên macOS

```bash
# 1) Cài Ollama (nếu chưa có)
brew install ollama
ollama serve &                     # hoặc mở app Ollama

# 2) Tải model (chọn model phù hợp cấu hình máy, càng nhỏ chạy CPU càng nhanh)
ollama pull llama3.2:1b             # model chat — có thể đổi sang qwen2.5:3b/phi3.5 nếu máy yếu
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
python cli.py ingest-faq ./data_raw/FAQ.xlsx
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

## 6. Streamlit demo app

Ngoài CLI, có sẵn 1 UI demo bằng Streamlit (`app_streamlit.py`) — phù hợp để demo trực tiếp cho
mentor mà không cần gõ lệnh:

```bash
pip install -r requirements.txt      # đã gồm streamlit, pandas
streamlit run app_streamlit.py
```

6 tab:

1. **Upload & Ingest** — upload trực tiếp `.docx`/`.xlsx` từ trình duyệt, xem report (inserted /
   exact_dup_skipped / near_dup_conflict / errors) + log ingest gần đây.
2. **Single QA Demo** — hỏi 1 câu, xem trả lời, nguồn, `cache_hit`, latency.
3. **Repeated Question / Stable Knowledge Reuse** — hỏi lặp lại (hoặc diễn đạt lại) 1 câu để
   thấy cache chuyển từ "chưa ổn định" sang "ổn định" (`is_stable=1`) và latency giảm hẳn ở các
   lượt sau, có biểu đồ latency theo từng lượt.
4. **Batch Questions** — upload CSV/XLSX cột `question`, chạy hàng loạt qua `QAService`, xem bảng
   `question / answer / sources / latency_sec / cache_hit / tier_used`, tải kết quả CSV.
5. **Memory Monitor** — thống kê theo status/tier/category, chạy phân tầng nóng/lạnh + TTL thủ
   công (có dry-run), xem log ghi vết nguồn gốc.
6. **Duplicate / Conflict Monitor** — xử lý các near-duplicate đang chờ (xem mục 6-conflict bên
   dưới): so sánh nội dung 2 bên, chọn "Dùng bản mới" / "Giữ bản cũ" / "Bỏ qua".

**Sidebar — Demo mode**: bật để hạ nhanh `STABLE_CACHE_MIN_HITS`, `HOT_ACCESS_THRESHOLD`,
`NEAR_DUP_THRESHOLD` (mutate trực tiếp `config` module lúc runtime, không cần sửa file/khởi động
lại) — giúp thấy hiệu ứng cache-hit / hot-tiering / conflict chỉ sau vài lượt tương tác thay vì
phải đợi dữ liệu tích luỹ nhiều ngày. Sidebar cũng cho chọn LLM backend (Ollama / llama.cpp, xem
mục 7) và nút xoá toàn bộ dữ liệu demo để chạy lại từ đầu.

### Conflict workflow (near-duplicate)

Trước đây near-duplicate bị tự động "supersede" bản cũ ngay khi ingest. Từ bản này, hành vi đổi
lại cho phù hợp demo/kiểm soát hơn:

- Khi `add_knowledge()` phát hiện 1 đoạn tri thức mới có cosine similarity ≥ `NEAR_DUP_THRESHOLD`
  so với 1 đoạn đang active cùng category, **cả 2 bản vẫn giữ `status='active'`** — không có gì
  bị ẩn/xoá tự động.
- 1 bản ghi được tạo trong bảng `conflicts` với `conflict_type='near_duplicate'`,
  `conflict_status='open'`.
- Người dùng xử lý qua UI (tab Duplicate/Conflict Monitor) hoặc gọi thẳng
  `MemoryStore.resolve_conflict(conflict_id, resolution)` với `resolution` là:
  - `"use_new"` — bản mới thắng, bản cũ chuyển `status='superseded'`.
  - `"keep_old"` — bản cũ thắng, bản mới chuyển `status='superseded'`.
  - `"ignored"` — coi là không phải xung đột thật, giữ nguyên cả 2 bản `active` song song.
- Không có bước LLM tự động giải quyết conflict — luôn cần xác nhận thủ công.
- `MemoryStore.list_conflicts(status=...)` và `MemoryStore.conflict_stats()` phục vụ UI/báo cáo.

## 7. Chạy bằng llama.cpp server (demo prefix/prompt cache thật)

Ollama không expose API công khai để thao tác trực tiếp KV-cache, nên bản gốc chỉ mô phỏng "tái
sử dụng trạng thái suy luận" bằng semantic answer cache. Backend `llamacpp` bổ sung ở đây gọi
thẳng `llama.cpp server` (`llama-server`), vốn có cơ chế **prefix-cache / prompt-cache thật** ở
mức KV-cache theo từng "slot".

### Cài & chạy llama-server (macOS)

```bash
brew install llama.cpp        # hoặc build từ https://github.com/ggml-org/llama.cpp

# Tiến trình phục vụ chat — nhớ set context đủ lớn cho system prompt + retrieved context
llama-server -m ./models/qwen2.5-7b-instruct-q4_k_m.gguf -c 8192 --port 8080

# (khuyến nghị) tiến trình RIÊNG phục vụ embedding, bật cờ --embeddings
llama-server -m ./models/nomic-embed-text-v1.5.Q8_0.gguf -c 2048 --port 8081 --embeddings
```

### Bật backend trong MemOS-lite

```bash
export MEMOS_LLM_BACKEND=llamacpp
export MEMOS_LLAMACPP_HOST=http://localhost:8080
python cli.py --backend llamacpp ask "Điều kiện vay tiền qua ViettelPay Pro là gì?"
```

Hoặc chọn `llamacpp` ở sidebar của `app_streamlit.py`. Nếu client chạy embedding trên tiến
trình/host riêng (khuyến nghị ở trên), truyền `embed_host` khi khởi tạo `LlamaCppClient` trực
tiếp trong code (hiện `client_factory.get_client()` dùng chung 1 host cho cả chat & embedding —
sửa nhanh nếu bạn tách 2 tiến trình).

### Cách prefix-cache hoạt động (và cách kiểm chứng)

- `llamacpp_client.py` gửi `cache_prompt: true` + `id_slot` **cố định** (mặc định 0, đổi qua
  `MEMOS_LLAMACPP_SLOT_ID`) trong mọi request `/v1/chat/completions`, để các lượt hỏi liên tiếp
  luôn route về cùng 1 slot — điều kiện bắt buộc để llama.cpp server tái sử dụng KV-cache của
  phần prefix trùng nhau (ở đây là `SYSTEM_PROMPT` cố định, và context nếu 2 câu hỏi liên tiếp
  cùng category nên có nhiều đoạn tri thức trùng nhau).
- Để xác nhận cache đang hoạt động: chạy `llama-server` với log mức mặc định và quan sát dòng
  log dạng `slot ... | reusing X tokens from cache` hoặc so sánh trực tiếp latency lượt 1 (cold,
  phải xử lý toàn bộ prompt) với lượt 2 hỏi câu tương tự cùng category (nên thấy giảm rõ rệt ở
  phần time-to-first-token).
- Nếu bản `llama.cpp` bạn đang chạy không hỗ trợ hoặc không cho hiệu quả cache rõ ràng (tham số
  server có thể đổi tên qua từng phiên bản), hệ thống vẫn **fallback an toàn** về semantic QA
  cache sẵn có trong `qa_service.py` — không có gì bị gãy, chỉ là không tận dụng được prefix-cache
  ở tầng thấp.

## 8. Giới hạn đã biết / phạm vi ngoài giai đoạn 1

- Hot/cold dựa trên luật đơn giản (access_count + thời gian), không dùng ML — đúng tinh thần
  "đơn giản" nhưng kém chính xác hơn nếu truy vấn dồn dập trong thời gian ngắn thay vì rải đều.
- Backend mặc định (`ollama`) vẫn chỉ mô phỏng "tái sử dụng trạng thái suy luận (KV)" bằng cache
  câu trả lời ngữ nghĩa, vì Ollama không expose API public để thao tác KV-cache thô. Backend
  `llamacpp` (mục 7) dùng cơ chế prefix-cache THẬT của llama.cpp server, nhưng hiệu quả phụ thuộc
  vào việc các câu hỏi liên tiếp có chung prefix (system prompt + context) hay không, và vào
  phiên bản llama.cpp đang chạy — nếu không đạt hiệu quả rõ ràng, hệ thống fallback về semantic
  cache như cũ (không có gì bị gãy).
- Near-dup dedup so sánh trong cùng category, ngưỡng cosine cố định (`NEAR_DUP_THRESHOLD`, có thể
  chỉnh qua sidebar demo). Từ bản này near-dup KHÔNG còn tự động supersede — được đưa vào hàng đợi
  `conflicts` để người dùng tự quyết định (mục 6) thay vì tự động đánh version/supersede như bản
  trước; vẫn chưa có phân quyền/kiểm toán ai xử lý conflict (nằm ở "Governance đầy đủ", ngoài
  phạm vi mục 4).
- Không có phân quyền, đa tenant, đa mô hình, multimodal (đúng yêu cầu đã loại khỏi MVP).
