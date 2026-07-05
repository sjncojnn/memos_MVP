"""
Cấu hình trung tâm cho MemOS-lite MVP.
Sửa các giá trị dưới đây (hoặc override qua biến môi trường) để phù hợp máy bạn.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DB_PATH = os.environ.get("MEMOS_DB_PATH", str(DATA_DIR / "memos.db"))
# DB RIÊNG cho RAG baseline (baseline_*.py) — tách biệt hoàn toàn với memos.db để so sánh công
# bằng: baseline không có dedup/tier/TTL/cache nên schema cũng đơn giản hơn nhiều (1 bảng
# `chunks` phẳng, xem baseline_db.py).
BASELINE_DB_PATH = os.environ.get("MEMOS_BASELINE_DB_PATH", str(DATA_DIR / "baseline.db"))

# ---- Ollama ----
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# Model chat: nên chọn model nhỏ chạy được CPU trên macOS, vd qwen2.5:7b, llama3.1:8b, phi3.5
CHAT_MODEL = os.environ.get("MEMOS_CHAT_MODEL", "llama3.2:3b")
# Model embedding: nomic-embed-text nhẹ, chạy tốt trên CPU
EMBED_MODEL = os.environ.get("MEMOS_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.environ.get("MEMOS_EMBED_DIM", "768"))  # nomic-embed-text = 768

# ---- Backend LLM có thể chọn: "ollama" (mặc định) hoặc "llamacpp" ----
# "llamacpp" dùng llama.cpp server để demo true prefix/prompt cache (xem README mục 7 và
# llamacpp_client.py). Chọn qua biến môi trường MEMOS_LLM_BACKEND hoặc sidebar Streamlit.
LLM_BACKEND = os.environ.get("MEMOS_LLM_BACKEND", "ollama")
LLAMACPP_HOST = os.environ.get("MEMOS_LLAMACPP_HOST", "http://localhost:8080")
# id_slot cố định để llama.cpp server luôn route về cùng 1 slot -> tối đa hoá khả năng
# cache prefix hit giữa các lượt hỏi liên tiếp (xem ghi chú trong llamacpp_client.py)
LLAMACPP_SLOT_ID = int(os.environ.get("MEMOS_LLAMACPP_SLOT_ID", "0"))

# ---- Ingest / dedup ----
CHUNK_MAX_CHARS = 1200          # kích thước tối đa 1 đơn vị tri thức (chunk)
CHUNK_OVERLAP_CHARS = 150
EXACT_DEDUP = True               # loại trùng lặp tuyệt đối theo hash nội dung
NEAR_DUP_THRESHOLD = 0.95        # cosine similarity >= ngưỡng này coi là gần trùng / xung đột phiên bản

# ---- Retrieval ----
TOP_K = 5
MIN_RESULTS_BEFORE_FALLBACK_COLD = 2   # nếu active/hot/warm không đủ kết quả, mới xét thêm 'cold'
SIM_THRESHOLD_MIN = 0.35               # bỏ qua kết quả có similarity quá thấp

# ---- Vòng đời / phân tầng nóng-lạnh (Section 4) ----
# LƯU Ý PHÂN TẦNG (Section 4) LÀ 1 TẦNG KHÁC, ĐỘC LẬP VỚI QA CACHE Ở DƯỚI:
#   - tier (hot/warm/cold) chỉ ảnh hưởng THỨ TỰ ƯU TIÊN RETRIEVAL trong
#     MemoryStore.search(): tier='cold' bị loại khỏi kết quả trừ khi không đủ ứng viên
#     hot/warm (xem fallback trong search()). Tier KHÔNG ảnh hưởng gì tới qa_cache.
#   - QA_CACHE_* bên dưới là tầng để BỎ QUA HOÀN TOÀN việc gọi LLM (kể cả retrieval) khi 1
#     câu hỏi/ý định đã "ổn định" — tức 2 cơ chế tối ưu hiệu năng khác nhau, không dùng
#     chung điều kiện hay ngưỡng.
DEFAULT_TTL_DAYS = None          # None = không hết hạn mặc định; đặt số ngày để bật TTL cho item mới
HOT_ACCESS_THRESHOLD = 5         # access_count trong cửa sổ HOT_WINDOW_DAYS để lên 'hot' (demo: có thể hạ xuống 2)
HOT_WINDOW_DAYS = 14
COLD_AFTER_DAYS = 30             # không truy cập > N ngày -> 'cold'
COLD_ACCESS_THRESHOLD = 2        # và access_count thấp hơn ngưỡng này mới bị đẩy 'cold'

# ---- Cache câu trả lời ổn định (mô phỏng tái sử dụng "trạng thái suy luận") ----
# GHI CHÚ DEMO: các giá trị dưới đây (STABLE_CACHE_MIN_HITS, HOT_ACCESS_THRESHOLD,
# QA_CACHE_MATCH_THRESHOLD) có thể bị GHI ĐÈ lúc runtime từ sidebar app_streamlit.py (mutate
# trực tiếp thuộc tính module `config`) để demo hiệu ứng cache/tiering nhanh trong vài lượt
# hỏi thay vì phải đợi nhiều ngày. Giá trị mặc định dưới đây dùng cho môi trường "thật".
#
# QUAN TRỌNG: đây là ngưỡng DUY NHẤT dùng để quyết định "2 câu hỏi có cùng ý định (intent)
# hay không" — dùng chung cho cả việc gộp hits (câu hỏi diễn đạt khác nhau nhưng cùng ý) LẪN
# việc trả lời tắt khi đã "stable". Trước đây bước tìm cache dùng min_sim=0.0 (luôn trả về
# ứng viên gần nhất dù không thực sự giống) -> gây cộng nhầm hits / ghi đè answer vào 1 cache
# row không liên quan. Đã sửa: chỉ coi là "khớp" khi similarity >= ngưỡng này.
# 0.85 phù hợp với embedding model biết ngữ nghĩa (nomic-embed-text...); nếu đổi sang model/
# embedding khác (đặc biệt embedding kiểu lexical/hash) cần đo lại và chỉnh ngưỡng cho phù hợp.
QA_CACHE_MATCH_THRESHOLD = 0.85
STABLE_CACHE_MIN_HITS = 3           # số lần hỏi trùng ý trước khi 1 cặp Q&A được coi là "ổn định" để cache

# ---- Budget / vòng đời cho qa_cache (tránh phình vô hạn) ----
# QA_CACHE_MAX_ITEMS: số dòng tối đa trong qa_cache; vượt ngưỡng -> xoá bớt các dòng lâu
# không được hỏi lại nhất (LRU theo last_hit_at). None = không giới hạn số lượng.
QA_CACHE_MAX_ITEMS = int(os.environ.get("MEMOS_QA_CACHE_MAX_ITEMS", "500"))
# QA_CACHE_TTL_DAYS: dòng cache không được hỏi lại quá N ngày sẽ bị xoá hẳn (không chỉ cold hoá
# như knowledge_units, vì 1 câu hỏi cache đã "chết" thì giữ lại không có giá trị). None = tắt.
QA_CACHE_TTL_DAYS = int(os.environ.get("MEMOS_QA_CACHE_TTL_DAYS", "30"))

# ---- Nhãn phân loại nghiệp vụ mặc định (tuỳ biến theo 16 file .docx thực tế) ----
KNOWN_CATEGORIES = [
    "tong_quan", "bao_hiem", "bhxh", "vay_tien", "nap_chuyen_tien",
    "cuoc_vien_thong", "hoa_don_dien_nuoc", "tai_chinh", "faq", "khac",
]
