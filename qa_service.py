"""
Pipeline hỏi-đáp (RAG) + cache câu trả lời ổn định.

Luồng xử lý mỗi câu hỏi:
  1. Kiểm tra qa_cache: nếu có câu hỏi đã cache với is_stable=1 và cosine similarity
     >= STABLE_CACHE_SIM_THRESHOLD -> trả lời NGAY, KHÔNG gọi LLM (latency ~ms).
     Đây là hiện thực cho yêu cầu "tái sử dụng tri thức ổn định, tần suất cao ... để
     giảm độ trễ" (Problem Statement mục 4) theo hướng cache ngữ nghĩa - lựa chọn thực tế
     vì Ollama không expose API để lưu/khôi phục KV-cache thô giữa các tiến trình.
  2. Nếu không trúng cache: retrieval qua MemoryStore.search() -> build prompt có
     system prompt CỐ ĐỊNH (giúp Ollama/llama.cpp tái sử dụng KV-cache nội bộ cho phần
     prefix chung) -> gọi model chat -> lưu vào qa_cache (hits=1, is_stable=0).
  3. Mỗi lần 1 câu hỏi tương tự lặp lại >= STABLE_CACHE_MIN_HITS lần -> đánh dấu is_stable=1,
     từ đó được dùng để trả lời tắt ở bước 1.
"""
import json
import uuid
from datetime import datetime

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
        with db.get_conn() as conn:
            rows = conn.execute(
                "SELECT id, answer, question_embedding, hits, is_stable, source_ids "
                "FROM qa_cache"
            ).fetchall()
            candidates = [(r["id"], vs.decode_embedding(r["question_embedding"])) for r in rows]
            if not candidates:
                return None
            hits = vs.top_k(query_vec, candidates, k=1, min_sim=0.0)
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

    def _store_cache(self, question: str, q_vec, answer: str, source_ids: list[str]):
        with db.get_conn() as conn:
            conn.execute(
                "INSERT INTO qa_cache (id, question_norm, question_embedding, answer, source_ids, "
                "hits, is_stable, created_at, last_hit_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), _normalize_q(question), vs.encode_embedding(q_vec), answer,
                 json.dumps(source_ids), 1, 0, _now(), _now())
            )

    # ------------------------------------------------------------ main
    def answer(self, question: str, top_k: int = None) -> dict:
        """
        Trả về dict:
          answer, sources[list of {source_file, source_ref, similarity}],
          cache_hit (bool), latency_sec, tier_used
        """
        t0 = datetime.utcnow()
        q_vec = self.client.embed(question)

        cache = self._find_cache_hit(q_vec)
        if cache and cache["sim"] >= config.STABLE_CACHE_SIM_THRESHOLD and cache["row"]["is_stable"]:
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

        source_ids = [r["id"] for r in results]
        if cache:
            # câu hỏi gần giống 1 câu đã hỏi trước nhưng chưa "stable" -> cộng dồn hits,
            # dùng lại answer mới nhất để cache luôn cập nhật theo tri thức mới nhất
            self._touch_cache(cache["row"]["id"], promote_stable=True)
            with db.get_conn() as conn:
                conn.execute("UPDATE qa_cache SET answer=?, source_ids=? WHERE id=?",
                             (answer_text, json.dumps(source_ids), cache["row"]["id"]))
        else:
            self._store_cache(question, q_vec, answer_text, source_ids)

        return {
            "answer": answer_text,
            "sources": [{"source_file": r["source_file"], "source_ref": r["source_ref"],
                         "similarity": r["similarity"]} for r in results],
            "cache_hit": False,
            "latency_sec": latency,
        }
