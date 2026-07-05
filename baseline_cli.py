"""
CLI cho RAG baseline — cấu trúc lệnh giống hệt cli.py (memos-mvp) để chạy song song, dễ so sánh.

Ví dụ:
    python baseline_cli.py init-db
    python baseline_cli.py ingest-docs ./data_raw/tai_lieu_nghiep_vu
    python baseline_cli.py ingest-faq ./data_raw/faq.xlsx
    python baseline_cli.py ask "Điều kiện vay tiền qua ViettelPay Pro là gì?"
    python baseline_cli.py stats
    python baseline_cli.py eval ./data_raw/golden_set.csv

Chọn backend khác Ollama (vd llama.cpp), đặt TRƯỚC subcommand, giống cli.py:
    python baseline_cli.py --backend llamacpp ask "..."
"""
import argparse

import config
import baseline_db as db
import client_factory
from baseline_store import BaselineStore
from baseline_qa import BaselineQAService
import baseline_ingest


def main():
    ap = argparse.ArgumentParser(description="RAG baseline CLI (đối chứng với memos-mvp)")
    ap.add_argument("--backend", choices=list(client_factory.BACKENDS), default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init-db")

    p_docs = sub.add_parser("ingest-docs")
    p_docs.add_argument("folder")

    p_faq = sub.add_parser("ingest-faq")
    p_faq.add_argument("path")

    p_ask = sub.add_parser("ask")
    p_ask.add_argument("question")

    sub.add_parser("stats")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("golden_csv")
    p_eval.add_argument("--no-judge", action="store_true")
    p_eval.add_argument("--out", default="eval_results_baseline.csv")

    args = ap.parse_args()

    if args.cmd == "init-db":
        db.init_db()
        print(f"OK - baseline database initialized at {config.BASELINE_DB_PATH}.")
        return

    db.init_db()  # idempotent, đảm bảo bảng tồn tại trước khi thao tác
    client = client_factory.get_client(backend=args.backend)
    store = BaselineStore(client)

    if args.cmd == "ingest-docs":
        summary = baseline_ingest.ingest_docx_folder(args.folder, store)
        print(summary)
    elif args.cmd == "ingest-faq":
        summary = baseline_ingest.ingest_faq_xlsx(args.path, store)
        print(summary)
    elif args.cmd == "ask":
        qa = BaselineQAService(client, store)
        result = qa.answer(args.question)
        print(f"\nTrả lời: {result['answer']}\n")
        print(f"Latency: {result['latency_sec']:.3f}s (baseline luôn gọi LLM, không cache)")
        for s in result["sources"]:
            print(f"  - [{s['similarity']:.3f}] {s['source_file']} | {s['source_ref']}")
    elif args.cmd == "stats":
        print(store.stats())
    elif args.cmd == "eval":
        from eval import run_eval  # tái sử dụng nguyên hàm tính EM/F1/judge của memos-mvp
        qa = BaselineQAService(client, store)
        summary = run_eval(args.golden_csv, qa, use_llm_judge=not args.no_judge, out_csv=args.out)
        for k, v in summary.items():
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
