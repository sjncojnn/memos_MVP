"""
Tự động hoá phân tầng nóng/lạnh & hết hạn TTL (Problem Statement mục 4 & 5).

Luật đơn giản, không cần ML, chạy được định kỳ (cron macOS `launchd`/`cron`) hoặc gọi
thủ công qua CLI (`python cli.py tier`):

  HOT   : access_count trong HOT_WINDOW_DAYS gần nhất >= HOT_ACCESS_THRESHOLD
          (đo gần đúng bằng last_accessed_at còn trong cửa sổ + access_count luỹ kế
          vượt ngưỡng, đủ cho MVP; muốn chính xác tuyệt đối cần thêm bảng access_log
          theo timestamp - để ngoài phạm vi giai đoạn 1)
  COLD  : last_accessed_at cũ hơn COLD_AFTER_DAYS VÀ access_count < COLD_ACCESS_THRESHOLD
  EXPIRE: ttl_expires_at đã qua -> status='expired' (loại khỏi mọi truy hồi)
  WARM  : còn lại (mặc định)

tier chỉ ảnh hưởng ĐỘ ƯU TIÊN truy hồi (search() ưu tiên hot/warm, cold là fallback);
KHÔNG xoá dữ liệu. TTL mới thực sự loại bỏ khỏi hệ thống (soft-delete qua status).
"""
from datetime import datetime, timedelta

import db
import config


def _now():
    return datetime.utcnow()


def run_tiering(dry_run: bool = False) -> dict:
    now = _now()
    hot_cutoff = (now - timedelta(days=config.HOT_WINDOW_DAYS)).isoformat()
    cold_cutoff = (now - timedelta(days=config.COLD_AFTER_DAYS)).isoformat()
    now_iso = now.isoformat()

    report = {"promoted_hot": 0, "demoted_cold": 0, "expired": 0, "reset_warm": 0}

    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT id, access_count, last_accessed_at, tier, status, ttl_expires_at "
            "FROM knowledge_units WHERE status='active'"
        ).fetchall()

        for r in rows:
            new_tier = r["tier"]
            if r["access_count"] >= config.HOT_ACCESS_THRESHOLD and \
                    r["last_accessed_at"] and r["last_accessed_at"] >= hot_cutoff:
                new_tier = "hot"
            elif (not r["last_accessed_at"] or r["last_accessed_at"] < cold_cutoff) and \
                    r["access_count"] < config.COLD_ACCESS_THRESHOLD:
                new_tier = "cold"
            else:
                new_tier = "warm"

            if new_tier != r["tier"]:
                if new_tier == "hot":
                    report["promoted_hot"] += 1
                elif new_tier == "cold":
                    report["demoted_cold"] += 1
                else:
                    report["reset_warm"] += 1
                if not dry_run:
                    conn.execute("UPDATE knowledge_units SET tier=? WHERE id=?", (new_tier, r["id"]))

        expired_rows = conn.execute(
            "SELECT id FROM knowledge_units WHERE status='active' AND ttl_expires_at IS NOT NULL "
            "AND ttl_expires_at < ?", (now_iso,)
        ).fetchall()
        report["expired"] = len(expired_rows)
        if not dry_run:
            for r in expired_rows:
                conn.execute("UPDATE knowledge_units SET status='expired', updated_at=? WHERE id=?",
                             (now_iso, r["id"]))

    return report


if __name__ == "__main__":
    print(run_tiering())
