"""
BaselineStore — "giao diện thống nhất" NHƯNG tối giản, đúng tinh thần RAG baseline thông thường:
add + search, không có update/delete/dedup/tier/TTL (khác hẳn MemoryStore trong memory_api.py).
"""
import uuid
from datetime import datetime

import baseline_db as db
import config
import vector_store as vs


def _now() -> str:
    return datetime.utcnow().isoformat()


class BaselineStore:
    def __init__(self, client):
        """client: đối tượng có .embed(text) -> np.ndarray (dùng chung OllamaClient/LlamaCppClient
        với memos-mvp để phép so sánh không bị lệch bởi khác biệt embedding model)."""
        self.client = client

    # ---------------------------------------------------------------- ADD
    def add_chunk(self, content: str, category: str = "khac", subcategory: str = None,
                  source_file: str = None, source_ref: str = None) -> dict:
        """KHÔNG kiểm tra trùng lặp dưới bất kỳ hình thức nào — mọi chunk được thêm thẳng vào
        DB, kể cả nội dung giống hệt chunk đã có. Đây là hành vi ĐIỂN HÌNH của 1 pipeline RAG
        thông thường không có bước quản lý vòng đời tri thức."""
        content = content.strip()
        if not content:
            return {"status": "error", "detail": "empty content"}
        embedding = self.client.embed(content)
        new_id = str(uuid.uuid4())
        with db.get_conn() as conn:
            conn.execute(
                "INSERT INTO chunks (id, content, category, subcategory, source_file, "
                "source_ref, created_at, embedding) VALUES (?,?,?,?,?,?,?,?)",
                (new_id, content, category, subcategory, source_file, source_ref, _now(),
                 vs.encode_embedding(embedding))
            )
            self._log(conn, source_file, "inserted", f"id={new_id[:8]}")
        return {"status": "inserted", "id": new_id}

    # ------------------------------------------------------------- SEARCH
    def search(self, query: str, top_k: int = None) -> list[dict]:
        """Cosine similarity brute-force trên TOÀN BỘ chunks — không có khái niệm hot/warm/cold,
        không ưu tiên gì cả, không cập nhật access_count (baseline không track usage)."""
        top_k = top_k or config.TOP_K
        query_vec = self.client.embed(query)
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT id, content, category, source_file, source_ref, embedding FROM chunks"
            ).fetchall()
        candidates = [(r["id"], vs.decode_embedding(r["embedding"])) for r in rows]
        hits = vs.top_k(query_vec, candidates, k=top_k, min_sim=config.SIM_THRESHOLD_MIN)
        by_id = {r["id"]: r for r in rows}
        results = []
        for hid, sim in hits:
            r = by_id[hid]
            results.append({
                "id": hid, "content": r["content"], "category": r["category"],
                "source_file": r["source_file"], "source_ref": r["source_ref"],
                "similarity": sim,
            })
        return results

    # -------------------------------------------------------------- STATS
    def stats(self) -> dict:
        with db.get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
            rows = conn.execute(
                "SELECT category, COUNT(*) c FROM chunks GROUP BY category"
            ).fetchall()
        return {"total": total, "by_category": {r["category"]: r["c"] for r in rows}}

    @staticmethod
    def _log(conn, source_file, status, detail):
        conn.execute(
            "INSERT INTO ingest_log (id, source_file, status, detail, created_at) "
            "VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), source_file or "", status, detail, _now())
        )
