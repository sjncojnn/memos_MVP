"""
Tiêu chí thành công (Problem Statement mục 5):
  - Chất lượng: LLM-judge (1-5) + F1/EM so với đáp án tham chiếu
  - Hiệu năng: độ trễ (latency), so sánh cache_hit vs cache_miss
  - Vòng đời: xem demo_lifecycle.py / scheduler.run_tiering()

Input: file CSV bộ đánh giá vàng với cột: question, reference_answer, topic (tuỳ chọn)
Output: results.csv (chi tiết từng câu) + summary in ra console
"""
import csv
import re
import statistics
from pathlib import Path

from qa_service import QAService


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text, flags=re.UNICODE)
    return text.split()


def exact_match(pred: str, ref: str) -> int:
    return int(_normalize(pred) == _normalize(ref))


def f1_score(pred: str, ref: str) -> float:
    pred_tokens, ref_tokens = _normalize(pred), _normalize(ref)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    for t, c in ref_counts.items():
        overlap += min(c, common.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def run_eval(golden_csv: str, qa: QAService, use_llm_judge: bool = True,
             out_csv: str = "eval_results.csv") -> dict:
    rows_out = []
    ems, f1s, judges, latencies_miss, latencies_hit = [], [], [], [], []

    with open(golden_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row["question"].strip()
            reference = row["reference_answer"].strip()
            topic = row.get("topic", "")

            result = qa.answer(question)
            pred = result["answer"]

            em = exact_match(pred, reference)
            f1 = f1_score(pred, reference)
            ems.append(em)
            f1s.append(f1)
            if result["cache_hit"]:
                latencies_hit.append(result["latency_sec"])
            else:
                latencies_miss.append(result["latency_sec"])

            judge_score = ""
            if use_llm_judge:
                try:
                    judge_score = qa.client.judge(question, reference, pred)
                    judges.append(judge_score)
                except Exception as e:  # noqa: BLE001
                    judge_score = f"error: {e}"

            rows_out.append({
                "question": question, "topic": topic, "reference": reference,
                "prediction": pred, "em": em, "f1": round(f1, 3),
                "judge_1_5": judge_score, "cache_hit": result["cache_hit"],
                "latency_sec": round(result["latency_sec"], 3),
            })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else [])
        writer.writeheader()
        writer.writerows(rows_out)

    summary = {
        "n": len(rows_out),
        "EM": round(statistics.mean(ems), 3) if ems else 0,
        "F1": round(statistics.mean(f1s), 3) if f1s else 0,
        "LLM_judge_avg_1_5": round(statistics.mean(judges), 2) if judges else None,
        "avg_latency_cache_miss_sec": round(statistics.mean(latencies_miss), 3) if latencies_miss else None,
        "avg_latency_cache_hit_sec": round(statistics.mean(latencies_hit), 3) if latencies_hit else None,
        "cache_hit_rate": round(len(latencies_hit) / len(rows_out), 3) if rows_out else 0,
        "results_file": str(Path(out_csv).resolve()),
    }
    return summary


if __name__ == "__main__":
    import argparse
    from ollama_client import OllamaClient

    ap = argparse.ArgumentParser()
    ap.add_argument("golden_csv")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--out", default="eval_results.csv")
    args = ap.parse_args()

    qa = QAService(OllamaClient())
    summary = run_eval(args.golden_csv, qa, use_llm_judge=not args.no_judge, out_csv=args.out)
    for k, v in summary.items():
        print(f"{k}: {v}")
