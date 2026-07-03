"""
Lớp lưu trữ SQLite cho MemOS-lite.

Bảng chính:
- knowledge_units : đơn vị tri thức đã chuẩn hoá (nội dung + nhãn + vòng đời + embedding)
- qa_cache        : cache câu hỏi-trả lời ổn định, tần suất cao (giả lập tái sử dụng "trạng thái suy luận")

Embedding được lưu dạng BLOB (numpy float32) ngay trong SQLite -> không cần dịch vụ vector rời
(đơn giản hoá so với MemOS gốc vốn dùng Neo4j/Qdrant). Có thể thay bằng Chroma sau này bằng cách
đổi VectorStore backend trong vector_store.py mà không đổi phần còn lại của hệ thống.
"""
import sqlite3
import contextlib
from pathlib import Path

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_units (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    category        TEXT NOT NULL DEFAULT 'khac',
    subcategory     TEXT,
    source_file     TEXT,
    source_ref      TEXT,
    content_hash    TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active',   -- active | cold | expired | superseded
    tier            TEXT NOT NULL DEFAULT 'warm',     -- hot | warm | cold
    version         INTEGER NOT NULL DEFAULT 1,
    superseded_by   TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count    INTEGER NOT NULL DEFAULT 0,
    ttl_expires_at  TEXT,
    embedding       BLOB
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_ku_hash ON knowledge_units(content_hash)
    WHERE status != 'expired';
CREATE INDEX IF NOT EXISTS idx_ku_status ON knowledge_units(status);
CREATE INDEX IF NOT EXISTS idx_ku_category ON knowledge_units(category);

CREATE TABLE IF NOT EXISTS qa_cache (
    id                TEXT PRIMARY KEY,
    question_norm     TEXT NOT NULL,
    question_embedding BLOB NOT NULL,
    answer            TEXT NOT NULL,
    source_ids        TEXT,          -- JSON list các knowledge_units.id đã dùng
    hits              INTEGER NOT NULL DEFAULT 1,
    is_stable         INTEGER NOT NULL DEFAULT 0,   -- 0/1, chỉ cache stable mới được dùng để trả lời tắt
    created_at        TEXT NOT NULL,
    last_hit_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingest_log (
    id           TEXT PRIMARY KEY,
    source_file  TEXT NOT NULL,
    status       TEXT NOT NULL,   -- inserted | exact_dup_skipped | near_dup_superseded | error
    detail       TEXT,
    created_at   TEXT NOT NULL
);
"""


@contextlib.contextmanager
def get_conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(SCHEMA)


if __name__ == "__main__":
    init_db()
    print(f"DB initialized at {config.DB_PATH}")
