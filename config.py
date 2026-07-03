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

# ---- Ollama ----
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
# Model chat: nên chọn model nhỏ chạy được CPU trên macOS, vd qwen2.5:7b, llama3.1:8b, phi3.5
CHAT_MODEL = os.environ.get("MEMOS_CHAT_MODEL", "qwen2.5:7b")
# Model embedding: nomic-embed-text nhẹ, chạy tốt trên CPU
EMBED_MODEL = os.environ.get("MEMOS_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.environ.get("MEMOS_EMBED_DIM", "768"))  # nomic-embed-text = 768

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
DEFAULT_TTL_DAYS = None          # None = không hết hạn mặc định; đặt số ngày để bật TTL cho item mới
HOT_ACCESS_THRESHOLD = 5         # access_count trong cửa sổ HOT_WINDOW_DAYS để lên 'hot'
HOT_WINDOW_DAYS = 14
COLD_AFTER_DAYS = 30             # không truy cập > N ngày -> 'cold'
COLD_ACCESS_THRESHOLD = 2        # và access_count thấp hơn ngưỡng này mới bị đẩy 'cold'

# ---- Cache câu trả lời ổn định (mô phỏng tái sử dụng "trạng thái suy luận") ----
STABLE_CACHE_SIM_THRESHOLD = 0.92   # câu hỏi mới đủ giống câu đã cache -> trả lời ngay, bỏ qua gọi LLM
STABLE_CACHE_MIN_HITS = 3           # số lần hỏi trùng ý trước khi 1 cặp Q&A được coi là "ổn định" để cache

# ---- Nhãn phân loại nghiệp vụ mặc định (tuỳ biến theo 16 file .docx thực tế) ----
KNOWN_CATEGORIES = [
    "tong_quan", "bao_hiem", "bhxh", "vay_tien", "nap_chuyen_tien",
    "cuoc_vien_thong", "hoa_don_dien_nuoc", "tai_chinh", "faq", "khac",
]
