"""
Vector store cực đơn giản: embedding lưu trực tiếp trong SQLite (bảng knowledge_units /
qa_cache, cột BLOB), tìm kiếm bằng cosine similarity trên numpy.

Lý do không dùng Chroma mặc định: với quy mô dữ liệu của bài toán (16 tài liệu nghiệp vụ +
1 bộ FAQ -> vài nghìn chunk), brute-force cosine trên numpy đủ nhanh (<50ms), không cần thêm
tiến trình / dependency ngoài numpy -> đúng tinh thần "giữ code simple" khi chạy local trên
macOS không GPU. Nếu dữ liệu lớn hơn nhiều, thay thế bằng ChromaDB chỉ cần viết lại 2 hàm
dưới đây (encode_embedding/search) mà không đụng tới phần còn lại của hệ thống.
"""
import numpy as np


def encode_embedding(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def decode_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def top_k(query_vec: np.ndarray, candidates: list[tuple[str, np.ndarray]], k: int = 5,
          min_sim: float = 0.0) -> list[tuple[str, float]]:
    """
    candidates: list[(id, vector)]
    Trả về list[(id, similarity)] đã sắp xếp giảm dần, giới hạn k, lọc theo min_sim.
    """
    scored = []
    for cid, vec in candidates:
        sim = cosine_sim(query_vec, vec)
        if sim >= min_sim:
            scored.append((cid, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
