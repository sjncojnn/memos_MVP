"""
Ingest cho RAG baseline — TÁI SỬ DỤNG nguyên phần parse/chunk/clean từ ingest.py (parse_docx,
parse_faq_xlsx, guess_category, clean_text) để 2 hệ thống nhận đúng cùng 1 cách chia chunk/làm
sạch dữ liệu đầu vào; khác biệt DUY NHẤT là store.add_chunk() ở đây không dedup/không gắn TTL.
"""
from pathlib import Path

from ingest import parse_docx, parse_faq_xlsx, guess_category  # tái sử dụng, không viết lại
from baseline_store import BaselineStore


def ingest_docx_folder(folder: Path, store: BaselineStore) -> dict:
    folder = Path(folder)
    summary = {"files": 0, "inserted": 0, "errors": []}
    for path in sorted(folder.glob("*.docx")):
        summary["files"] += 1
        category = guess_category(path.name)
        try:
            sections = parse_docx(path)
        except Exception as e:  # noqa: BLE001
            summary["errors"].append(f"{path.name}: {e}")
            continue
        for sec in sections:
            res = store.add_chunk(
                content=sec["content"], category=category, source_file=path.name,
                source_ref=sec["source_ref"],
            )
            if res["status"] != "inserted":
                summary["errors"].append(f"{path.name}: {res.get('detail')}")
            else:
                summary["inserted"] += 1
    return summary


def ingest_faq_xlsx(path: Path, store: BaselineStore) -> dict:
    path = Path(path)
    summary = {"files": 1, "inserted": 0, "errors": []}
    try:
        items = parse_faq_xlsx(path)
    except Exception as e:  # noqa: BLE001
        summary["errors"].append(f"{path.name}: {e}")
        return summary
    for item in items:
        res = store.add_chunk(
            content=item["content"], category="faq", subcategory=item.get("subcategory"),
            source_file=path.name, source_ref=item["source_ref"],
        )
        if res["status"] != "inserted":
            summary["errors"].append(f"{path.name}: {res.get('detail')}")
        else:
            summary["inserted"] += 1
    return summary
