"""
Pipeline hỏi-đáp cho RAG BASELINE: retrieval + gọi LLM ở MỌI câu hỏi, không có bất kỳ tầng
tối ưu nào (không QA cache, không hot/cold, không "trạng thái ổn định"). Đây là đối trọng để
so sánh với qa_service.QAService của memos-mvp (mục 5 Problem Statement: "vượt baseline truy
hồi tri thức thông thường").

Trả về dict CÙNG SHAPE với QAService.answer() (answer, sources, cache_hit, latency_sec) — luôn
cache_hit=False — để tái sử dụng nguyên hàm eval.run_eval() cho cả 2 hệ thống mà không cần viết
lại logic tính EM/F1/LLM-judge (xem eval_compare.py).
"""
from datetime import datetime

from baseline_store import BaselineStore


class BaselineQAService:
    def __init__(self, client, store: BaselineStore = None):
        self.client = client
        self.store = store or BaselineStore(client)

    def answer(self, question: str, top_k: int = None) -> dict:
        t0 = datetime.utcnow()
        results = self.store.search(question, top_k=top_k)
        context = "\n\n".join(
            f"[{r['source_file']} | {r['source_ref']}]\n{r['content']}" for r in results
        ) or "(không tìm thấy tri thức liên quan)"

        gen = self.client.chat(question, context=context)
        latency = gen["latency_sec"]

        return {
            "answer": gen["answer"],
            "sources": [{"id": r["id"], "source_file": r["source_file"],
                         "source_ref": r["source_ref"], "similarity": r["similarity"]}
                        for r in results],
            "cache_hit": False,   # baseline không có khái niệm cache -> luôn False
            "latency_sec": latency,
        }
