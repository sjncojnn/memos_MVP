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
          - near-dup theo cosine similarity trong cùng category -> KHÔNG tự động
            supersede; thay vào đó tạo 1 bản ghi trong bảng `conflicts` (status='open')
            để người dùng tự xử lý (xem list_conflicts/resolve_conflict). Cả bản cũ và
            bản mới vẫn 'active' cho tới khi conflict được xử lý.
        Trả về dict {status, id, conflict_id} với status in
          {inserted, exact_dup_skipped, near_dup_conflict}
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
            # LƯU Ý: khác với bản trước, ở đây KHÔNG tự động supersede bản cũ nữa.
            # Nếu phát hiện gần trùng, cả 2 bản (cũ + mới) vẫn 'active' song song, và một
            # bản ghi conflict (status='open') được tạo để người dùng tự quyết định qua
            # UI/API (xem list_conflicts / resolve_conflict) — đúng tinh thần "không tự động
            # xóa/ẩn bản cũ nếu chưa cần" của quy trình conflict workflow.
            conflict_old_id = None
            conflict_sim = None
            rows = conn.execute(
                "SELECT id, embedding FROM knowledge_units WHERE category=? AND status='active'",
                (category,)
            ).fetchall()
            candidates = [(r["id"], vs.decode_embedding(r["embedding"])) for r in rows if r["embedding"]]
            if candidates:
                best = vs.top_k(embedding, candidates, k=1, min_sim=config.NEAR_DUP_THRESHOLD)
                if best:
                    conflict_old_id, conflict_sim = best[0]

            new_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO knowledge_units (id, content, category, subcategory, source_file, "
                "source_ref, content_hash, status, tier, version, created_at, updated_at, "
                "access_count, ttl_expires_at, embedding) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (new_id, content, category, subcategory, source_file, source_ref, content_hash,
                 "active", "warm", 1, now, now, 0, ttl_expires_at,
                 vs.encode_embedding(embedding))
            )

            conflict_id = None
            if conflict_old_id:
                conflict_id = str(uuid.uuid4())
                conn.execute(
                    "INSERT INTO conflicts (id, old_id, new_id, category, similarity, "
                    "conflict_type, conflict_status, created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (conflict_id, conflict_old_id, new_id, category, conflict_sim,
                     "near_duplicate", "open", now)
                )
                status = "near_dup_conflict"
            else:
                status = "inserted"

            detail = f"id={new_id[:8]}"
            if conflict_old_id:
                detail += f" conflict_with={conflict_old_id[:8]} sim={conflict_sim:.3f}"
            self._log(conn, source_file, status, detail)
            return {"status": status, "id": new_id, "conflict_id": conflict_id}

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

    # ----------------------------------------------------------- CONFLICTS
    def list_conflicts(self, status: str | None = "open") -> list[dict]:
        """
        Liệt kê conflict kèm nội dung 2 bên (old/new) để hiển thị trong UI.
        status=None -> lấy tất cả trạng thái (open/resolved_use_new/resolved_keep_old/ignored).
        """
        with db.get_conn() as conn:
            sql = (
                "SELECT c.id, c.old_id, c.new_id, c.category, c.similarity, c.conflict_type, "
                "c.conflict_status, c.created_at, c.resolved_at, "
                "ou.content AS old_content, ou.source_file AS old_source_file, "
                "ou.source_ref AS old_source_ref, ou.status AS old_status, ou.version AS old_version, "
                "nu.content AS new_content, nu.source_file AS new_source_file, "
                "nu.source_ref AS new_source_ref, nu.status AS new_status, nu.version AS new_version "
                "FROM conflicts c "
                "JOIN knowledge_units ou ON c.old_id = ou.id "
                "JOIN knowledge_units nu ON c.new_id = nu.id"
            )
            params = []
            if status:
                sql += " WHERE c.conflict_status=?"
                params.append(status)
            sql += " ORDER BY c.created_at DESC"
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    def resolve_conflict(self, conflict_id: str, resolution: str) -> dict:
        """
        Xử lý 1 conflict. resolution in {'use_new', 'keep_old', 'ignored'}:
          - use_new  : bản mới thắng -> supersede bản cũ (bản cũ status='superseded'), version
                       của bản mới += 1 kế thừa từ bản cũ.
          - keep_old : bản cũ thắng -> bản mới bị supersede (status='superseded'), bản cũ giữ nguyên.
          - ignored  : bỏ qua, giữ nguyên cả 2 bản 'active' song song (coi là không xung đột thật).
        Không có bước LLM tự động giải quyết — luôn cần người dùng xác nhận.
        """
        valid = {"use_new", "keep_old", "ignored"}
        if resolution not in valid:
            return {"status": "error", "detail": f"resolution phải thuộc {sorted(valid)}"}
        now = _now()
        with db.get_conn() as conn:
            row = conn.execute("SELECT * FROM conflicts WHERE id=?", (conflict_id,)).fetchone()
            if not row:
                return {"status": "not_found"}
            if row["conflict_status"] != "open":
                return {"status": "already_resolved", "conflict_status": row["conflict_status"]}

            old_id, new_id = row["old_id"], row["new_id"]
            if resolution == "use_new":
                old_row = conn.execute("SELECT version FROM knowledge_units WHERE id=?",
                                        (old_id,)).fetchone()
                new_version = (old_row["version"] if old_row else 1) + 1
                conn.execute(
                    "UPDATE knowledge_units SET status='superseded', superseded_by=?, updated_at=? "
                    "WHERE id=?", (new_id, now, old_id)
                )
                conn.execute(
                    "UPDATE knowledge_units SET version=?, updated_at=? WHERE id=?",
                    (new_version, now, new_id)
                )
                conflict_status = "resolved_use_new"
            elif resolution == "keep_old":
                conn.execute(
                    "UPDATE knowledge_units SET status='superseded', superseded_by=?, updated_at=? "
                    "WHERE id=?", (old_id, now, new_id)
                )
                conflict_status = "resolved_keep_old"
            else:  # ignored
                conflict_status = "ignored"

            conn.execute(
                "UPDATE conflicts SET conflict_status=?, resolved_at=? WHERE id=?",
                (conflict_status, now, conflict_id)
            )
            return {"status": conflict_status, "conflict_id": conflict_id}

    def conflict_stats(self) -> dict:
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT conflict_status, COUNT(*) c FROM conflicts GROUP BY conflict_status"
            ).fetchall()
            return {r["conflict_status"]: r["c"] for r in rows}

    @staticmethod
    def _log(conn, source_file, status, detail):
        conn.execute(
            "INSERT INTO ingest_log (id, source_file, status, detail, created_at) VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), source_file or "", status, detail, _now())
        )
