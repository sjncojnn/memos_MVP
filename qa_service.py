"""
Pipeline hỏi-đáp (RAG) + cache câu trả lời ổn định.

Đây là 1 TẦNG TỐI ƯU RIÊNG, độc lập với hot/cold tier của knowledge_units (tier chỉ ảnh
hưởng thứ tự ưu tiên trong MemoryStore.search(), xem config.py). QA cache hoạt động ở mức
"câu hỏi/ý định", không phải ở mức tri thức: mục tiêu là khi 1 ý định hỏi đã lặp lại đủ
nhiều lần (ổn định), bỏ qua HOÀN TOÀN cả retrieval lẫn gọi LLM.

Luồng xử lý mỗi câu hỏi:
  1. Tìm trong qa_cache 1 câu hỏi đã lưu có cosine similarity >= QA_CACHE_MATCH_THRESHOLD
     với câu hỏi mới (áp dụng NGƯỠNG THỰC SỰ — khác bản trước hay trả về "ứng viên gần nhất"
     dù similarity rất thấp, gây cộng nhầm hits/ghi đè answer sai). Đây cũng là cách hệ
     thống nhận diện các câu hỏi ĐỒNG NGHĨA/cùng ý định dù khác cách diễn đạt, miễn embedding
     model đủ tốt về ngữ nghĩa (nomic-embed-text trở lên).
  2a. Nếu tìm thấy VÀ đã is_stable=1 -> trả lời NGAY từ cache, KHÔNG gọi LLM (latency ~ms).
  2b. Nếu tìm thấy nhưng CHƯA stable -> vẫn gọi LLM (để câu trả lời luôn đúng/mới nhất cho
     tới khi đủ tin cậy), đồng thời cộng hits vào ĐÚNG cache row đó (không tạo row mới).
  2c. Nếu KHÔNG tìm thấy match nào đủ ngưỡng -> gọi LLM, tạo cache row MỚI (hits=1).
  3. Mỗi lần 1 câu hỏi/ý định lặp lại >= STABLE_CACHE_MIN_HITS lần -> đánh dấu is_stable=1.
  4. Sau mỗi lần tạo cache row mới, áp dụng budget (QA_CACHE_MAX_ITEMS, QA_CACHE_TTL_DAYS) để
     cache không phình vô hạn — xem _enforce_cache_limits().

Vì cùng đi qua đúng 1 hàm answer() này, Single QA / Batch Questions dùng chung 1 logic cache
-> không còn khác biệt hành vi giữa các chỗ gọi, miễn cùng 1 câu hỏi/ý định sẽ luôn khớp vào
cùng 1 cache row bất kể được hỏi từ đâu.
"""
import json
import uuid
from datetime import datetime, timedelta

import db
import config
import vector_store as vs
from memory_api import MemoryStore, _now


def _normalize_q(q: str) -> str:
    return " ".join(q.strip().lower().split())


class QAService:
    def __init__(self, client, store: MemoryStore = None):
        self.client = client
        self.store = store or MemoryStore(client)

    # ------------------------------------------------------------ cache
    def _find_cache_hit(self, query_vec) -> dict | None:
        """Tìm cache row có cosine similarity CAO NHẤT với query_vec, nhưng CHỈ coi là khớp
        (trả về khác None) nếu similarity >= config.QA_CACHE_MATCH_THRESHOLD. Trước đây hàm
        này dùng min_sim=0.0 nên luôn trả về 1 "ứng viên" dù chẳng liên quan gì -> khiến các
        câu hỏi hoàn toàn khác nhau bị gộp nhầm vào cùng 1 cache row (bug đã sửa)."""
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT id, answer, question_embedding, hits, is_stable, source_ids "
                "FROM qa_cache"
            ).fetchall()
            candidates = [(r["id"], vs.decode_embedding(r["question_embedding"])) for r in rows]
            if not candidates:
                return None
            hits = vs.top_k(query_vec, candidates, k=1, min_sim=config.QA_CACHE_MATCH_THRESHOLD)
            if not hits:
                return None
            best_id, sim = hits[0]
            row = next(r for r in rows if r["id"] == best_id)
            return {"row": row, "sim": sim}

    def _touch_cache(self, cache_id: str, promote_stable: bool):
        with db.get_conn() as conn:
            if promote_stable:
                conn.execute(
                    "UPDATE qa_cache SET hits=hits+1, last_hit_at=?, "
                    "is_stable=CASE WHEN hits+1>=? THEN 1 ELSE is_stable END WHERE id=?",
                    (_now(), config.STABLE_CACHE_MIN_HITS, cache_id)
                )
            else:
                conn.execute("UPDATE qa_cache SET hits=hits+1, last_hit_at=? WHERE id=?",
                             (_now(), cache_id))

    def _store_cache(self, question: str, q_vec, answer: str, sources_meta: list[dict]):
        with db.get_conn() as conn:
            conn.execute(
                "INSERT INTO qa_cache (id, question_norm, question_embedding, answer, source_ids, "
                "hits, is_stable, created_at, last_hit_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), _normalize_q(question), vs.encode_embedding(q_vec), answer,
                 json.dumps(sources_meta), 1, 0, _now(), _now())
            )
            self._enforce_cache_limits(conn)

    def _enforce_cache_limits(self, conn):
        """Budget cho qa_cache (điểm 5): TTL trước, rồi LRU theo last_hit_at nếu vẫn còn vượt
        QA_CACHE_MAX_ITEMS. Gọi sau mỗi lần INSERT dòng mới (không gọi ở mỗi cache-hit để đỡ
        overhead — cache-hit chỉ UPDATE, không làm tăng số dòng)."""
        if config.QA_CACHE_TTL_DAYS:
            cutoff = (datetime.utcnow() - timedelta(days=config.QA_CACHE_TTL_DAYS)).isoformat()
            conn.execute("DELETE FROM qa_cache WHERE last_hit_at < ?", (cutoff,))
        if config.QA_CACHE_MAX_ITEMS:
            total = conn.execute("SELECT COUNT(*) c FROM qa_cache").fetchone()["c"]
            excess = total - config.QA_CACHE_MAX_ITEMS
            if excess > 0:
                old_ids = conn.execute(
                    "SELECT id FROM qa_cache ORDER BY last_hit_at ASC LIMIT ?", (excess,)
                ).fetchall()
                conn.executemany("DELETE FROM qa_cache WHERE id=?", [(r["id"],) for r in old_ids])

    def enforce_cache_limits_now(self) -> dict:
        """Wrapper public để UI (Memory Monitor) có thể dọn cache thủ công theo yêu cầu."""
        with db.get_conn() as conn:
            before = conn.execute("SELECT COUNT(*) c FROM qa_cache").fetchone()["c"]
            self._enforce_cache_limits(conn)
            after = conn.execute("SELECT COUNT(*) c FROM qa_cache").fetchone()["c"]
        return {"removed": before - after, "remaining": after}

    def cache_stats(self) -> dict:
        with db.get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) c FROM qa_cache").fetchone()["c"]
            stable = conn.execute("SELECT COUNT(*) c FROM qa_cache WHERE is_stable=1").fetchone()["c"]
        return {
            "total": total, "stable": stable, "not_stable": total - stable,
            "max_items": config.QA_CACHE_MAX_ITEMS, "ttl_days": config.QA_CACHE_TTL_DAYS,
        }

    def cache_info(self, question: str) -> dict | None:
        """Đọc trạng thái cache hiện tại (hits, is_stable) của ĐÚNG câu hỏi này theo chuẩn hoá
        chính xác (không qua similarity) — dùng để hiển thị tiến trình tới STABLE_CACHE_MIN_HITS
        trong tab Single QA Demo. Lưu ý: đây tra theo question_norm chính xác của CÂU HỎI NÀY,
        khác với _find_cache_hit() tra theo similarity nên có thể trỏ tới 1 câu hỏi diễn đạt
        khác (đồng nghĩa) đã tồn tại trước đó."""
        qn = _normalize_q(question)
        with db.get_conn() as conn:
            row = conn.execute(
                "SELECT hits, is_stable, last_hit_at FROM qa_cache WHERE question_norm=?", (qn,)
            ).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------ main
    def answer(self, question: str, top_k: int = None) -> dict:
        """
        Trả về dict:
          answer, sources[list of {source_file, source_ref, similarity}],
          cache_hit (bool), latency_sec, tier_used
        """
        t0 = datetime.utcnow()
        q_vec = self.client.embed(question)

        # _find_cache_hit() giờ chỉ trả về khác None khi similarity >= QA_CACHE_MATCH_THRESHOLD
        # (xem docstring) -> "cache" ở đây LUÔN là 1 match thật sự, không còn khả năng là 1
        # row không liên quan bị chọn nhầm.
        cache = self._find_cache_hit(q_vec)

        # Nhánh 1: đã "ổn định" (hỏi đủ nhiều lần cùng ý định) -> trả lời tắt, KHÔNG gọi LLM.
        if cache and cache["row"]["is_stable"]:
            self._touch_cache(cache["row"]["id"], promote_stable=False)
            latency = (datetime.utcnow() - t0).total_seconds()
            return {
                "answer": cache["row"]["answer"],
                "sources": json.loads(cache["row"]["source_ids"] or "[]"),
                "cache_hit": True,
                "latency_sec": latency,
            }

        results = self.store.search(question, top_k=top_k)
        context = "\n\n".join(
            f"[{r['source_file']} | {r['source_ref']}]\n{r['content']}" for r in results
        ) or "(không tìm thấy tri thức liên quan)"

        gen = self.client.chat(question, context=context)
        answer_text = gen["answer"]
        latency = gen["latency_sec"]

        # Định dạng sources THỐNG NHẤT dù trả lời từ cache hay từ LLM mới -> dùng chung cho
        # UI (Batch Questions, Memory/Conflict Monitor) suy ra tier_used/nguồn mà không cần
        # phân biệt 2 nhánh.
        sources_meta = [
            {"id": r["id"], "source_file": r["source_file"], "source_ref": r["source_ref"],
             "tier": r.get("tier"), "similarity": r.get("similarity")}
            for r in results
        ]
        if cache:
            # Nhánh 2: cùng ý định với 1 cache row đã tồn tại nhưng CHƯA stable -> cộng hits
            # vào ĐÚNG row đó (không tạo row mới, không đụng tới row khác), đồng thời cập nhật
            # answer/sources theo câu trả lời mới nhất để cache luôn phản ánh tri thức mới nhất.
            self._touch_cache(cache["row"]["id"], promote_stable=True)
            with db.get_conn() as conn:
                conn.execute("UPDATE qa_cache SET answer=?, source_ids=? WHERE id=?",
                             (answer_text, json.dumps(sources_meta), cache["row"]["id"]))
        else:
            # Nhánh 3: không có ý định nào tương tự trong cache -> tạo entry mới (hits=1).
            self._store_cache(question, q_vec, answer_text, sources_meta)

        return {
            "answer": answer_text,
            "sources": sources_meta,
            "cache_hit": False,
            "latency_sec": latency,
        }
