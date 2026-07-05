"""
Lớp lưu trữ cho RAG BASELINE — dùng để so sánh với memos-mvp (mục 5 Problem Statement:
"vượt baseline truy hồi tri thức thông thường").

CHỦ ĐÍCH đơn giản hơn nhiều so với db.py của memos-mvp, vì đây là baseline mô phỏng cách làm
RAG "thông thường" — KHÔNG có:
  - exact/near-dup dedup (mọi chunk ingest đều được thêm thẳng, kể cả trùng lặp)
  - status/tier (không hot/cold, không phân tầng, mọi chunk được tìm kiếm như nhau)
  - version/superseded (không quản lý phiên bản, không conflict workflow)
  - TTL/vòng đời (chunk tồn tại vĩnh viễn, không hết hạn)
  - QA cache (mỗi câu hỏi luôn retrieval + gọi LLM lại từ đầu, không có "trạng thái ổn định")
Đây chính là các trục khác biệt mà memos-mvp cải thiện — giữ baseline "thuần" để phép so sánh
(eval_compare.py) đo được đúng giá trị gia tăng của phần quản lý bộ nhớ.

Dùng CÙNG cơ chế lưu embedding (BLOB numpy qua vector_store.py) và CÙNG LLM backend
(client_factory.py) với memos-mvp để biến số DUY NHẤT giữa 2 hệ thống là "có/không có quản lý
bộ nhớ", không phải khác biệt về model hay cách encode.
"""
import sqlite3
import contextlib
from pathlib import Path

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    content      TEXT NOT NULL,
    category     TEXT NOT NULL DEFAULT 'khac',
    subcategory  TEXT,
    source_file  TEXT,
    source_ref   TEXT,
    created_at   TEXT NOT NULL,
    embedding    BLOB
);
CREATE INDEX IF NOT EXISTS idx_chunks_category ON chunks(category);

CREATE TABLE IF NOT EXISTS ingest_log (
    id           TEXT PRIMARY KEY,
    source_file  TEXT NOT NULL,
    status       TEXT NOT NULL,   -- inserted | error  (KHÔNG có exact_dup_skipped/near_dup_conflict)
    detail       TEXT,
    created_at   TEXT NOT NULL
);
"""


@contextlib.contextmanager
def get_conn():
    conn = sqlite3.connect(config.BASELINE_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    Path(config.BASELINE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA)


if __name__ == "__main__":
    init_db()
    print(f"Baseline DB initialized at {config.BASELINE_DB_PATH}")
