"""
So sánh memos-mvp vs RAG baseline trên CÙNG 1 bộ đánh giá vàng (Problem Statement mục 5:
"vượt baseline truy hồi tri thức thông thường — đánh giá bằng LLM-judge kèm chỉ số định lượng
F1/EM").

QUAN TRỌNG để so sánh công bằng: chạy `cli.py` (ingest cho memos) và `baseline_cli.py` (ingest
cho baseline) trên CÙNG bộ dữ liệu thô trước khi chạy script này, và dùng CÙNG --backend/model
cho cả 2 hệ thống (script này chỉ nhận 1 giá trị --backend áp dụng cho cả 2 phía).

Dùng lại nguyên hàm eval.run_eval() cho cả 2 phía (baseline_qa.BaselineQAService.answer() trả
về cùng shape dict với qa_service.QAService.answer()), nên số liệu EM/F1/LLM-judge/latency được
tính bằng đúng 1 công thức, không lệch do cách đo khác nhau.

Ví dụ:
    python eval_compare.py ./data_raw/golden_set.csv
    python eval_compare.py ./data_raw/golden_set.csv --backend llamacpp --no-judge
"""
import argparse
import statistics

import client_factory
from eval import run_eval
from qa_service import QAService
from memory_api import MemoryStore
from baseline_qa import BaselineQAService
from baseline_store import BaselineStore
import db
import baseline_db


def _fmt(v):
    if v is None or v == "":
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def compare(golden_csv: str, backend: str = None, use_llm_judge: bool = True,
            memos_out: str = "eval_results_memos.csv",
            baseline_out: str = "eval_results_baseline.csv") -> dict:
    db.init_db()
    baseline_db.init_db()

    client = client_factory.get_client(backend=backend)

    memos_qa = QAService(client, MemoryStore(client))
    baseline_qa = BaselineQAService(client, BaselineStore(client))

    print(f"=== Chạy memos-mvp trên {golden_csv} ===")
    memos_summary = run_eval(golden_csv, memos_qa, use_llm_judge=use_llm_judge, out_csv=memos_out)

    print(f"\n=== Chạy RAG baseline trên {golden_csv} ===")
    baseline_summary = run_eval(golden_csv, baseline_qa, use_llm_judge=use_llm_judge, out_csv=baseline_out)

    rows = [
        ("EM", memos_summary["EM"], baseline_summary["EM"]),
        ("F1", memos_summary["F1"], baseline_summary["F1"]),
        ("LLM_judge_avg_1_5", memos_summary["LLM_judge_avg_1_5"], baseline_summary["LLM_judge_avg_1_5"]),
        ("avg_latency_cache_miss_sec", memos_summary["avg_latency_cache_miss_sec"],
         baseline_summary["avg_latency_cache_miss_sec"]),
        ("avg_latency_cache_hit_sec", memos_summary["avg_latency_cache_hit_sec"],
         baseline_summary["avg_latency_cache_hit_sec"]),
        ("cache_hit_rate", memos_summary["cache_hit_rate"], baseline_summary["cache_hit_rate"]),
    ]

    print("\n" + "=" * 72)
    print(f"{'Chỉ số':<28}{'memos-mvp':>18}{'baseline':>18}")
    print("-" * 72)
    for name, m_val, b_val in rows:
        print(f"{name:<28}{_fmt(m_val):>18}{_fmt(b_val):>18}")
    print("=" * 72)
    print(
        "Ghi chú: baseline không có QA cache -> avg_latency_cache_miss_sec của baseline là "
        "latency TRUNG BÌNH THỰC SỰ của mọi câu hỏi (luôn phải gọi LLM); "
        "avg_latency_cache_hit_sec/cache_hit_rate của baseline luôn '-'/0 vì không có khái niệm "
        "cache. Đây chính là điểm baseline THIẾU mà memos-mvp bổ sung (mục 4 'Tối ưu hiệu năng')."
    )

    return {"memos": memos_summary, "baseline": baseline_summary}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="So sánh memos-mvp vs RAG baseline")
    ap.add_argument("golden_csv")
    ap.add_argument("--backend", choices=list(client_factory.BACKENDS), default=None,
                     help="Áp dụng CHUNG cho cả 2 hệ thống để so sánh công bằng")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--memos-out", default="eval_results_memos.csv")
    ap.add_argument("--baseline-out", default="eval_results_baseline.csv")
    args = ap.parse_args()

    compare(args.golden_csv, backend=args.backend, use_llm_judge=not args.no_judge,
            memos_out=args.memos_out, baseline_out=args.baseline_out)
