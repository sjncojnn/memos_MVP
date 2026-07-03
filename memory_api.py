"""
Giao diện thống nhất quản lý tri thức (Problem Statement mục 4):
  "Truy hồi & giao diện thống nhất cho thêm/tìm/cập nhật tri thức"

Class MemoryStore là entrypoint DUY NHẤT mà các phần khác (ingest, qa_service, scheduler,
eval) dùng để chạm vào dữ liệu -> đảm bảo mọi thao tác đi qua cùng 1 chỗ để dễ audit,
đổi backend (vd SQLite -> Postgres, hoặc thêm Chroma) mà không phải sửa logic nghiệp vụ.

client (OllamaClient hoặc tương đương) được TIÊM VÀO (dependency injection) để test offline
bằng fake client, không cần Ollama server thật.
"""
import hashlib
import uuid
from datetime import datetime, timedelta

import db
import config
import vector_store as vs


def _now() -> str:
    return datetime.utcnow().isoformat()


def _hash_content(text: str) -> str:
    normalized = " ".join(text.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class MemoryStore:
    def __init__(self, client):
        """client: đối tượng có .embed(text) -> np.ndarray (OllamaClient hoặc FakeOllamaClient)."""
        self.client = client

    # ---------------------------------------------------------------- ADD
    def add_knowledge(self, content: str, category: str = "khac", subcategory: str = None,
                       source_file: str = None, source_ref: str = None,
                       ttl_days: int = None) -> dict:
        """
        Thêm 1 đơn vị tri thức. Có kiểm tra:
          - exact dedup theo content_hash (bỏ qua nếu đã tồn tại, còn active)
          - near-dup theo cosine similarity trong cùng category (đánh dấu superseded bản cũ,
            tăng version) -> đáp ứng "khử trùng lặp" + gợi ý quản lý phiên bản tri thức (mục 3 note)
        Trả về dict {status, id} với status in
          {inserted, exact_dup_skipped, near_dup_superseded}
        """
        content = content.strip()
        if not content:
            return {"status": "error", "detail": "empty content"}

        content_hash = _hash_content(content)
        now = _now()
        ttl_expires_at = None
        if ttl_days:
            ttl_expires_at = (datetime.utcnow() + timedelta(days=ttl_days)).isoformat()

        with db.get_conn() as conn:
            if config.EXACT_DEDUP:
                row = conn.execute(
                    "SELECT id FROM knowledge_units WHERE content_hash=? AND status!='expired'",
                    (content_hash,)
                ).fetchone()
                if row:
                    self._log(conn, source_file, "exact_dup_skipped", f"hash={content_hash[:12]}")
                    return {"status": "exact_dup_skipped", "id": row["id"]}

            embedding = self.client.embed(content)

            # near-duplicate check trong cùng category (giới hạn phạm vi so sánh cho gọn & nhanh)
            superseded_id = None
            rows = conn.execute(
                "SELECT id, embedding FROM knowledge_units WHERE category=? AND status='active'",
                (category,)
            ).fetchall()
            candidates = [(r["id"], vs.decode_embedding(r["embedding"])) for r in rows if r["embedding"]]
            if candidates:
                best = vs.top_k(embedding, candidates, k=1, min_sim=config.NEAR_DUP_THRESHOLD)
                if best:
                    superseded_id = best[0][0]

            new_id = str(uuid.uuid4())
            version = 1
            if superseded_id:
                old = conn.execute("SELECT version FROM knowledge_units WHERE id=?",
                                    (superseded_id,)).fetchone()
                version = (old["version"] if old else 1) + 1
                conn.execute(
                    "UPDATE knowledge_units SET status='superseded', superseded_by=?, updated_at=? "
                    "WHERE id=?", (new_id, now, superseded_id)
                )

            conn.execute(
                "INSERT INTO knowledge_units (id, content, category, subcategory, source_file, "
                "source_ref, content_hash, status, tier, version, created_at, updated_at, "
                "access_count, ttl_expires_at, embedding) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (new_id, content, category, subcategory, source_file, source_ref, content_hash,
                 "active", "warm", version, now, now, 0, ttl_expires_at,
                 vs.encode_embedding(embedding))
            )
            status = "near_dup_superseded" if superseded_id else "inserted"
            self._log(conn, source_file, status, f"id={new_id[:8]} version={version}")
            return {"status": status, "id": new_id}

    # ------------------------------------------------------------- SEARCH
    def search(self, query: str, top_k: int = None, categories: list[str] = None,
               include_cold_fallback: bool = True) -> list[dict]:
        """
        Tìm các knowledge_units liên quan nhất tới query. Mặc định chỉ tìm trong
        status='active' (bao gồm tier hot/warm); nếu không đủ kết quả mới fallback
        sang tier='cold' (vẫn status active, chỉ là "lưu trữ nguội").
        Mỗi lần match sẽ tăng access_count + cập nhật last_accessed_at (phục vụ tiering).
        """
        top_k = top_k or config.TOP_K
        query_vec = self.client.embed(query)

        with db.get_conn() as conn:
            sql = "SELECT id, content, category, source_file, source_ref, tier, embedding " \
                  "FROM knowledge_units WHERE status='active' AND tier!='cold'"
            params = []
            if categories:
                sql += " AND category IN (%s)" % ",".join("?" * len(categories))
                params += categories
            rows = conn.execute(sql, params).fetchall()
            candidates = [(r["id"], vs.decode_embedding(r["embedding"])) for r in rows]
            hits = vs.top_k(query_vec, candidates, k=top_k, min_sim=config.SIM_THRESHOLD_MIN)

            by_id = {r["id"]: r for r in rows}
            if include_cold_fallback and len(hits) < config.MIN_RESULTS_BEFORE_FALLBACK_COLD:
                sql_cold = "SELECT id, content, category, source_file, source_ref, tier, embedding " \
                           "FROM knowledge_units WHERE status='active' AND tier='cold'"
                rows_cold = conn.execute(sql_cold).fetchall()
                for r in rows_cold:
                    by_id[r["id"]] = r
                cand_cold = [(r["id"], vs.decode_embedding(r["embedding"])) for r in rows_cold]
                hits += vs.top_k(query_vec, cand_cold, k=top_k, min_sim=config.SIM_THRESHOLD_MIN)
                hits.sort(key=lambda x: x[1], reverse=True)
                hits = hits[:top_k]

            results = []
            now = _now()
            for hid, sim in hits:
                r = by_id[hid]
                conn.execute(
                    "UPDATE knowledge_units SET access_count=access_count+1, last_accessed_at=? "
                    "WHERE id=?", (now, hid)
                )
                results.append({
                    "id": hid, "content": r["content"], "category": r["category"],
                    "source_file": r["source_file"], "source_ref": r["source_ref"],
                    "similarity": sim, "tier": r["tier"],
                })
            return results

    # ------------------------------------------------------------- UPDATE
    def update_knowledge(self, unit_id: str, new_content: str = None, category: str = None,
                          ttl_days: int = None) -> dict:
        with db.get_conn() as conn:
            row = conn.execute("SELECT * FROM knowledge_units WHERE id=?", (unit_id,)).fetchone()
            if not row:
                return {"status": "not_found"}
            now = _now()
            content = new_content if new_content is not None else row["content"]
            cat = category if category is not None else row["category"]
            embedding = vs.decode_embedding(row["embedding"])
            content_hash = row["content_hash"]
            if new_content is not None:
                embedding = self.client.embed(new_content)
                content_hash = _hash_content(new_content)
            ttl_expires_at = row["ttl_expires_at"]
            if ttl_days is not None:
                ttl_expires_at = (datetime.utcnow() + timedelta(days=ttl_days)).isoformat()
            conn.execute(
                "UPDATE knowledge_units SET content=?, category=?, content_hash=?, embedding=?, "
                "ttl_expires_at=?, version=version+1, updated_at=? WHERE id=?",
                (content, cat, content_hash, vs.encode_embedding(embedding), ttl_expires_at, now, unit_id)
            )
            return {"status": "updated", "id": unit_id}

    # ------------------------------------------------------------- DELETE
    def delete_knowledge(self, unit_id: str, hard: bool = False) -> dict:
        with db.get_conn() as conn:
            if hard:
                conn.execute("DELETE FROM knowledge_units WHERE id=?", (unit_id,))
            else:
                conn.execute("UPDATE knowledge_units SET status='expired', updated_at=? WHERE id=?",
                             (_now(), unit_id))
            return {"status": "deleted", "id": unit_id, "hard": hard}

    def get_by_id(self, unit_id: str) -> dict | None:
        with db.get_conn() as conn:
            row = conn.execute("SELECT * FROM knowledge_units WHERE id=?", (unit_id,)).fetchone()
            return dict(row) if row else None

    def stats(self) -> dict:
        with db.get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) c FROM knowledge_units").fetchone()["c"]
            by_status = conn.execute(
                "SELECT status, COUNT(*) c FROM knowledge_units GROUP BY status").fetchall()
            by_tier = conn.execute(
                "SELECT tier, COUNT(*) c FROM knowledge_units WHERE status='active' GROUP BY tier"
            ).fetchall()
            by_category = conn.execute(
                "SELECT category, COUNT(*) c FROM knowledge_units WHERE status='active' "
                "GROUP BY category").fetchall()
            return {
                "total": total,
                "by_status": {r["status"]: r["c"] for r in by_status},
                "by_tier": {r["tier"]: r["c"] for r in by_tier},
                "by_category": {r["category"]: r["c"] for r in by_category},
            }

    @staticmethod
    def _log(conn, source_file, status, detail):
        conn.execute(
            "INSERT INTO ingest_log (id, source_file, status, detail, created_at) VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), source_file or "", status, detail, _now())
        )
