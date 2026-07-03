"""
CLI duy nhất cho toàn bộ MVP.

Ví dụ:
    python cli.py init-db
    python cli.py ingest-docs ./data_raw/tai_lieu_nghiep_vu
    python cli.py ingest-faq ./data_raw/faq.xlsx
    python cli.py ask "Điều kiện vay tiền qua ViettelPay Pro là gì?"
    python cli.py tier                 # chạy phân tầng nóng/lạnh + TTL
    python cli.py tier --dry-run
    python cli.py stats
    python cli.py eval ./data_raw/golden_set.csv

Chọn LLM backend khác Ollama (vd llama.cpp server, xem README mục 7) bằng --backend, LƯU Ý
đặt TRƯỚC subcommand (giới hạn của argparse subparsers):
    python cli.py --backend llamacpp ask "..."
"""
import argparse
import json
import sys

import db
from memory_api import MemoryStore
from qa_service import QAService
import client_factory
import ingest
import scheduler


def main():
    ap = argparse.ArgumentParser(description="MemOS-lite MVP CLI")
    ap.add_argument("--backend", choices=list(client_factory.BACKENDS), default=None,
                     help="Chọn LLM backend: ollama (mặc định) hoặc llamacpp")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db", help="Khởi tạo database SQLite")

    p_docs = sub.add_parser("ingest-docs", help="Ingest thư mục chứa file .docx nghiệp vụ")
    p_docs.add_argument("folder")
    p_docs.add_argument("--ttl-days", type=int, default=None)

    p_faq = sub.add_parser("ingest-faq", help="Ingest file FAQ .xlsx")
    p_faq.add_argument("path")
    p_faq.add_argument("--ttl-days", type=int, default=None)

    p_ask = sub.add_parser("ask", help="Hỏi 1 câu")
    p_ask.add_argument("question")

    p_tier = sub.add_parser("tier", help="Chạy phân tầng nóng/lạnh + hết hạn TTL")
    p_tier.add_argument("--dry-run", action="store_true")

    sub.add_parser("stats", help="Thống kê kho tri thức")

    p_eval = sub.add_parser("eval", help="Chạy bộ đánh giá vàng (CSV: question,reference_answer,topic)")
    p_eval.add_argument("golden_csv")
    p_eval.add_argument("--no-judge", action="store_true")
    p_eval.add_argument("--out", default="eval_results.csv")

    args = ap.parse_args()

    if args.cmd == "init-db":
        db.init_db()
        print("OK - database initialized.")
        return

    # các lệnh còn lại cần LLM backend đang chạy (Ollama hoặc llama.cpp server)
    client = client_factory.get_client(backend=args.backend)
    store = MemoryStore(client)

    if args.cmd == "ingest-docs":
        summary = ingest.ingest_docx_folder(args.folder, store, ttl_days=args.ttl_days)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    elif args.cmd == "ingest-faq":
        summary = ingest.ingest_faq_xlsx(args.path, store, ttl_days=args.ttl_days)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    elif args.cmd == "ask":
        qa = QAService(client, store)
        result = qa.answer(args.question)
        print("\n=== TRẢ LỜI ===")
        print(result["answer"])
        print(f"\n(cache_hit={result['cache_hit']}, latency={result['latency_sec']:.2f}s)")
        if result["sources"]:
            print("\nNguồn:")
            for s in result["sources"]:
                print(f"  - {s.get('source_file')} | {s.get('source_ref')} "
                      f"(sim={s.get('similarity', 0):.2f})")

    elif args.cmd == "tier":
        report = scheduler.run_tiering(dry_run=args.dry_run)
        print(json.dumps(report, ensure_ascii=False, indent=2))

    elif args.cmd == "stats":
        print(json.dumps(store.stats(), ensure_ascii=False, indent=2))

    elif args.cmd == "eval":
        import eval as eval_mod
        qa = QAService(client, store)
        summary = eval_mod.run_eval(args.golden_csv, qa, use_llm_judge=not args.no_judge, out_csv=args.out)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"Lỗi: {e}", file=sys.stderr)
        sys.exit(1)
