"""
Streamlit demo UI cho MemOS-lite MVP.

Chạy:
    streamlit run app_streamlit.py

6 tab theo yêu cầu demo:
  1. Upload & Ingest                       -> ingest .docx/.xlsx từ UI, xem report
  2. Single QA Demo                        -> hỏi 1 câu, xem answer/sources/latency/cache
  3. Repeated Question / Stable Reuse      -> hỏi lặp lại để thấy cache "ổn định" kích hoạt
  4. Batch Questions                       -> upload CSV/XLSX câu hỏi, chạy hàng loạt, tải CSV
  5. Memory Monitor                        -> thống kê kho tri thức + chạy phân tầng thủ công
  6. Duplicate / Conflict Monitor          -> xử lý near-duplicate (use new / keep old / ignore)

Sidebar: chọn LLM backend (Ollama / llama.cpp), bật "Demo mode" để hạ nhanh
STABLE_CACHE_MIN_HITS / HOT_ACCESS_THRESHOLD / NEAR_DUP_THRESHOLD giúp thấy hiệu ứng
cache-hit & hot/cold tiering trong vài lượt hỏi thay vì phải đợi nhiều ngày.
"""
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

import config
import db
import client_factory
import ingest
import scheduler
from memory_api import MemoryStore
from qa_service import QAService

st.set_page_config(page_title="MemOS-lite Demo", layout="wide")
db.init_db()


# ----------------------------------------------------------------- helpers
def _get_store_and_qa():
    """Tạo/lấy lại MemoryStore + QAService theo backend đang chọn trong sidebar, cache trong
    session_state để không phải tạo lại client ở mỗi lần rerun (Streamlit rerun toàn bộ script
    mỗi khi có tương tác)."""
    backend = st.session_state.get("backend", config.LLM_BACKEND)
    cache_key = f"_client_{backend}"
    if cache_key not in st.session_state:
        try:
            st.session_state[cache_key] = client_factory.get_client(backend=backend)
        except Exception as e:  # noqa: BLE001
            st.error(f"Không khởi tạo được client cho backend '{backend}': {e}")
            st.stop()
    client = st.session_state[cache_key]

    if st.session_state.get("_store_backend") != backend:
        st.session_state["_store"] = MemoryStore(client)
        st.session_state["_qa"] = QAService(client, st.session_state["_store"])
        st.session_state["_store_backend"] = backend

    return st.session_state["_store"], st.session_state["_qa"]


def _save_uploads(uploaded_files, target_dir: Path) -> list[Path]:
    paths = []
    for f in uploaded_files:
        p = target_dir / f.name
        with open(p, "wb") as out:
            out.write(f.getbuffer())
        paths.append(p)
    return paths


def _merge_summary(total: dict, part: dict):
    for k, v in part.items():
        if k == "errors":
            total.setdefault("errors", []).extend(v)
        elif isinstance(v, int):
            total[k] = total.get(k, 0) + v


def _sources_to_text(sources: list[dict]) -> str:
    if not sources:
        return ""
    return "; ".join(
        f"{s.get('source_file') or ''}|{s.get('source_ref') or ''}" for s in sources
    )


def _tier_used(sources: list[dict]) -> str:
    tiers = sorted({s.get("tier") for s in (sources or []) if s.get("tier")})
    return ",".join(tiers)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("⚙️ Cấu hình demo")

backend = st.sidebar.selectbox(
    "LLM backend", options=list(client_factory.BACKENDS),
    index=list(client_factory.BACKENDS).index(config.LLM_BACKEND)
    if config.LLM_BACKEND in client_factory.BACKENDS else 0,
    help="ollama = mặc định. llamacpp = llama-server, demo true prefix/prompt cache (mục 7 README).",
)
st.session_state["backend"] = backend
if backend == "llamacpp":
    st.sidebar.caption(f"llama-server host: `{config.LLAMACPP_HOST}` (đổi qua env MEMOS_LLAMACPP_HOST)")
else:
    st.sidebar.caption(f"Ollama host: `{config.OLLAMA_HOST}`")

st.sidebar.divider()
demo_mode = st.sidebar.checkbox(
    "🚀 Demo mode (cache/tiering nhanh)", value=True,
    help="Hạ ngưỡng để thấy hiệu ứng cache-hit / hot-tiering chỉ sau vài lượt hỏi.",
)
if demo_mode:
    config.STABLE_CACHE_MIN_HITS = st.sidebar.slider(
        "STABLE_CACHE_MIN_HITS", min_value=1, max_value=5, value=2,
        help="Số lần hỏi trùng ý trước khi cache được coi là 'ổn định' (trả lời tắt, bỏ qua LLM).",
    )
    config.HOT_ACCESS_THRESHOLD = st.sidebar.slider(
        "HOT_ACCESS_THRESHOLD", min_value=1, max_value=5, value=2,
        help="access_count để 1 knowledge unit được lên tier 'hot' khi chạy phân tầng.",
    )
    config.NEAR_DUP_THRESHOLD = st.sidebar.slider(
        "NEAR_DUP_THRESHOLD (cosine)", min_value=0.80, max_value=0.99, value=0.90, step=0.01,
        help="Ngưỡng để 2 đoạn tri thức bị coi là near-duplicate -> tạo conflict chờ xử lý.",
    )
else:
    config.STABLE_CACHE_MIN_HITS = 3
    config.HOT_ACCESS_THRESHOLD = 5
    config.NEAR_DUP_THRESHOLD = 0.95
st.sidebar.caption(
    f"Hiện tại: STABLE_CACHE_MIN_HITS={config.STABLE_CACHE_MIN_HITS} · "
    f"HOT_ACCESS_THRESHOLD={config.HOT_ACCESS_THRESHOLD} · "
    f"NEAR_DUP_THRESHOLD={config.NEAR_DUP_THRESHOLD:.2f}"
)

st.sidebar.divider()
with st.sidebar.expander("🗑️ Reset dữ liệu demo"):
    st.caption("Xoá toàn bộ knowledge_units / qa_cache / conflicts / ingest_log trong DB hiện tại.")
    confirm = st.checkbox("Tôi hiểu thao tác này không thể hoàn tác", key="confirm_reset")
    if st.button("Xoá toàn bộ dữ liệu", disabled=not confirm):
        with db.get_conn() as conn:
            conn.execute("DELETE FROM knowledge_units")
            conn.execute("DELETE FROM qa_cache")
            conn.execute("DELETE FROM conflicts")
            conn.execute("DELETE FROM ingest_log")
        st.success("Đã xoá dữ liệu. Tải lại trang / chuyển tab để cập nhật.")

store, qa = _get_store_and_qa()

st.title("🧠 MemOS-lite — Demo quản lý bộ nhớ cho LLM")

tabs = st.tabs([
    "📥 Upload & Ingest",
    "💬 Single QA Demo",
    "🔁 Repeated Question",
    "📊 Batch Questions",
    "🗄️ Memory Monitor",
    "⚠️ Duplicate / Conflict Monitor",
])

# ============================================================== TAB 1: Ingest
with tabs[0]:
    st.subheader("Upload tài liệu nghiệp vụ (.docx) và FAQ (.xlsx)")
    ttl_input = st.number_input(
        "TTL (số ngày, để 0 = không hết hạn)", min_value=0, value=0, step=1,
        help="Nếu > 0, các knowledge unit ingest lần này sẽ tự hết hạn (status=expired) sau N ngày.",
    )
    ttl_days = ttl_input or None

    col_docx, col_xlsx = st.columns(2)
    with col_docx:
        docx_files = st.file_uploader(
            "File .docx nghiệp vụ (có thể chọn nhiều)", type=["docx"], accept_multiple_files=True,
            key="docx_uploader",
        )
    with col_xlsx:
        xlsx_files = st.file_uploader(
            "File FAQ .xlsx (có thể chọn nhiều)", type=["xlsx"], accept_multiple_files=True,
            key="xlsx_uploader",
        )

    if st.button("▶️ Ingest", type="primary", disabled=not (docx_files or xlsx_files)):
        summary = {"files": 0, "inserted": 0, "exact_dup_skipped": 0, "near_dup_conflict": 0, "errors": []}
        with st.spinner("Đang ingest..."):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                if docx_files:
                    docx_dir = tmp_path / "docx"
                    docx_dir.mkdir()
                    _save_uploads(docx_files, docx_dir)
                    part = ingest.ingest_docx_folder(docx_dir, store, ttl_days=ttl_days)
                    _merge_summary(summary, part)
                if xlsx_files:
                    xlsx_dir = tmp_path / "xlsx"
                    xlsx_dir.mkdir()
                    xlsx_paths = _save_uploads(xlsx_files, xlsx_dir)
                    for p in xlsx_paths:
                        part = ingest.ingest_faq_xlsx(p, store, ttl_days=ttl_days)
                        _merge_summary(summary, part)

        st.success("Ingest hoàn tất.")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("File đã đọc", summary.get("files", 0))
        m2.metric("Inserted", summary.get("inserted", 0))
        m3.metric("Exact dup skipped", summary.get("exact_dup_skipped", 0))
        m4.metric("Near-dup / conflict", summary.get("near_dup_conflict", 0))
        m5.metric("Lỗi", len(summary.get("errors", [])))

        if summary.get("near_dup_conflict"):
            st.warning(
                f"Có {summary['near_dup_conflict']} conflict mới (near-duplicate) đang chờ xử lý "
                "→ xem tab **⚠️ Duplicate / Conflict Monitor**."
            )
        if summary.get("errors"):
            st.error("Lỗi khi ingest:")
            for e in summary["errors"]:
                st.code(e)

    st.divider()
    st.subheader("Log ingest gần đây")
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT source_file, status, detail, created_at FROM ingest_log "
            "ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
    if rows:
        st.dataframe(pd.DataFrame([dict(r) for r in rows]), use_container_width=True, hide_index=True)
    else:
        st.caption("Chưa có log ingest nào.")

# ============================================================== TAB 2: Single QA
with tabs[1]:
    st.subheader("Hỏi 1 câu")
    question = st.text_input("Câu hỏi nghiệp vụ", key="single_qa_question",
                              placeholder="VD: Điều kiện vay tiền qua ViettelPay Pro là gì?")
    if st.button("Hỏi", type="primary", key="single_qa_btn", disabled=not question):
        with st.spinner("Đang xử lý..."):
            try:
                result = qa.answer(question)
            except Exception as e:  # noqa: BLE001
                st.error(f"Lỗi khi gọi LLM backend ({backend}): {e}")
                result = None
        if result:
            st.markdown("**Trả lời:**")
            st.write(result["answer"])
            c1, c2 = st.columns(2)
            c1.metric("Cache hit", "✅ Có" if result["cache_hit"] else "❌ Không")
            c2.metric("Latency (s)", f"{result['latency_sec']:.3f}")
            if result["sources"]:
                st.markdown("**Nguồn tri thức đã dùng:**")
                st.dataframe(pd.DataFrame(result["sources"]), use_container_width=True, hide_index=True)
            else:
                st.caption("Không có nguồn tri thức nào được retrieve (có thể do kho tri thức trống).")

# ============================================================== TAB 3: Repeated Question
with tabs[2]:
    st.subheader("Hỏi lặp lại để xem cache 'ổn định' kích hoạt")
    st.caption(
        "Nhập cùng 1 câu hỏi (hoặc diễn đạt lại gần giống) và bấm 'Hỏi lại' nhiều lần. "
        f"Sau {config.STABLE_CACHE_MIN_HITS} lượt hỏi trùng ý, cache chuyển sang 'stable' và các lượt "
        "sau sẽ trả lời tức thì (cache_hit=True, không gọi LLM)."
    )
    repeat_q = st.text_input("Câu hỏi để lặp lại", key="repeat_question",
                              placeholder="VD: Phí giao dịch chuyển tiền liên ngân hàng là bao nhiêu?")

    if "repeat_history" not in st.session_state:
        st.session_state["repeat_history"] = []

    col_ask, col_clear = st.columns([1, 1])
    if col_ask.button("🔁 Hỏi lại (mô phỏng lặp lại)", type="primary", disabled=not repeat_q):
        with st.spinner("Đang xử lý..."):
            try:
                result = qa.answer(repeat_q)
                cache_info = qa.cache_info(repeat_q)
            except Exception as e:  # noqa: BLE001
                st.error(f"Lỗi khi gọi LLM backend ({backend}): {e}")
                result, cache_info = None, None
        if result:
            st.session_state["repeat_history"].append({
                "lượt": len(st.session_state["repeat_history"]) + 1,
                "question": repeat_q,
                "cache_hit": result["cache_hit"],
                "latency_sec": round(result["latency_sec"], 3),
                "hits": cache_info["hits"] if cache_info else None,
                "is_stable": bool(cache_info["is_stable"]) if cache_info else False,
            })
    if col_clear.button("Xoá lịch sử lượt hỏi (chỉ trong UI)"):
        st.session_state["repeat_history"] = []

    history = st.session_state["repeat_history"]
    if history:
        df_hist = pd.DataFrame(history)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
        st.markdown("**Latency theo từng lượt hỏi (giây):**")
        st.line_chart(df_hist.set_index("lượt")["latency_sec"])
        last = history[-1]
        if last["is_stable"]:
            st.success(
                f"✅ Cache đã 'ổn định' (hits={last['hits']} ≥ {config.STABLE_CACHE_MIN_HITS}) — "
                "các lượt hỏi tương tự tiếp theo sẽ được trả lời tức thì, bỏ qua LLM."
            )
        else:
            hits = last["hits"] or 0
            st.info(
                f"Đang tích luỹ: {hits}/{config.STABLE_CACHE_MIN_HITS} lượt trùng ý. "
                "Hỏi lại (đúng câu hoặc diễn đạt gần giống) để tiếp tục tích luỹ."
            )
    else:
        st.caption("Chưa có lượt hỏi nào trong phiên này.")

# ============================================================== TAB 4: Batch Questions
with tabs[3]:
    st.subheader("Chạy hàng loạt câu hỏi từ file")
    st.caption("File CSV hoặc XLSX cần có cột tên 'question' (không phân biệt hoa/thường).")
    batch_file = st.file_uploader("Upload CSV/XLSX câu hỏi", type=["csv", "xlsx"], key="batch_uploader")

    if batch_file is not None:
        try:
            if batch_file.name.lower().endswith(".csv"):
                df_in = pd.read_csv(batch_file)
            else:
                df_in = pd.read_excel(batch_file)
        except Exception as e:  # noqa: BLE001
            st.error(f"Không đọc được file: {e}")
            df_in = None

        if df_in is not None:
            col_map = {c.lower().strip(): c for c in df_in.columns}
            if "question" not in col_map:
                st.error("Không tìm thấy cột 'question' trong file.")
            else:
                q_col = col_map["question"]
                st.write(f"Tìm thấy {len(df_in)} câu hỏi.")
                st.dataframe(df_in[[q_col]].head(10), use_container_width=True, hide_index=True)

                if st.button("▶️ Chạy batch", type="primary"):
                    results = []
                    progress = st.progress(0.0, text="Đang xử lý...")
                    n = len(df_in)
                    for i, q in enumerate(df_in[q_col].astype(str)):
                        try:
                            r = qa.answer(q)
                            results.append({
                                "question": q,
                                "answer": r["answer"],
                                "sources": _sources_to_text(r["sources"]),
                                "latency_sec": round(r["latency_sec"], 3),
                                "cache_hit": r["cache_hit"],
                                "tier_used": _tier_used(r["sources"]),
                            })
                        except Exception as e:  # noqa: BLE001
                            results.append({
                                "question": q, "answer": f"[LỖI] {e}", "sources": "",
                                "latency_sec": None, "cache_hit": False, "tier_used": "",
                            })
                        progress.progress((i + 1) / n, text=f"Đang xử lý... {i + 1}/{n}")
                    progress.empty()

                    df_out = pd.DataFrame(results)
                    st.success(f"Đã xử lý xong {n} câu hỏi.")
                    st.dataframe(df_out, use_container_width=True, hide_index=True)

                    c1, c2 = st.columns(2)
                    c1.metric("Cache hit rate", f"{df_out['cache_hit'].mean() * 100:.1f}%")
                    valid_latency = df_out["latency_sec"].dropna()
                    c2.metric("Latency trung bình (s)",
                              f"{valid_latency.mean():.3f}" if len(valid_latency) else "-")

                    csv_bytes = df_out.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "⬇️ Tải kết quả CSV", data=csv_bytes,
                        file_name="batch_qa_results.csv", mime="text/csv",
                    )

# ============================================================== TAB 5: Memory Monitor
with tabs[4]:
    st.subheader("Thống kê kho tri thức")
    stats = store.stats()
    m1, m2 = st.columns(2)
    m1.metric("Tổng số knowledge unit", stats["total"])
    m2.metric("Đang active", stats["by_status"].get("active", 0))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Theo status**")
        st.dataframe(pd.DataFrame(
            [{"status": k, "count": v} for k, v in stats["by_status"].items()]
        ), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("**Theo tier (active)**")
        df_tier = pd.DataFrame([{"tier": k, "count": v} for k, v in stats["by_tier"].items()])
        st.dataframe(df_tier, use_container_width=True, hide_index=True)
        if not df_tier.empty:
            st.bar_chart(df_tier.set_index("tier")["count"])
    with c3:
        st.markdown("**Theo category (active)**")
        st.dataframe(pd.DataFrame(
            [{"category": k, "count": v} for k, v in stats["by_category"].items()]
        ), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Phân tầng nóng/lạnh & hết hạn TTL")
    dry_run = st.checkbox("Dry-run (chỉ xem trước, không cập nhật DB)", value=True)
    if st.button("⚙️ Chạy phân tầng ngay", type="primary"):
        report = scheduler.run_tiering(dry_run=dry_run)
        st.json(report)
        if not dry_run:
            st.success("Đã cập nhật tier/status trong DB.")
            st.rerun()

    st.divider()
    st.subheader("Ghi vết nguồn gốc (ingest_log gần đây)")
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT source_file, status, detail, created_at FROM ingest_log "
            "ORDER BY created_at DESC LIMIT 100"
        ).fetchall()
    st.dataframe(pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame(),
                 use_container_width=True, hide_index=True)

# ============================================================== TAB 6: Conflict Monitor
with tabs[5]:
    st.subheader("Duplicate / Conflict Monitor")
    st.caption(
        "Khi ingest phát hiện 2 đoạn tri thức gần giống nhau (near-duplicate), hệ thống KHÔNG tự "
        "động xoá/ẩn bản cũ — cả 2 vẫn 'active' song song và được đưa vào hàng đợi này để bạn tự "
        "quyết định. Không có bước LLM tự động giải quyết conflict."
    )

    cstats = store.conflict_stats()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Đang mở (open)", cstats.get("open", 0))
    m2.metric("Đã dùng bản mới", cstats.get("resolved_use_new", 0))
    m3.metric("Đã giữ bản cũ", cstats.get("resolved_keep_old", 0))
    m4.metric("Đã bỏ qua", cstats.get("ignored", 0))

    status_filter = st.selectbox(
        "Lọc theo trạng thái", ["open", "resolved_use_new", "resolved_keep_old", "ignored", "Tất cả"],
        index=0,
    )
    conflicts = store.list_conflicts(status=None if status_filter == "Tất cả" else status_filter)

    if not conflicts:
        st.info("Không có conflict nào ở trạng thái này. 🎉")

    for c in conflicts:
        sim_txt = f"{c['similarity']:.3f}" if c["similarity"] is not None else "-"
        header = (
            f"[{c['conflict_type']}] category={c['category']} · similarity={sim_txt} · "
            f"trạng thái={c['conflict_status']} · tạo lúc {c['created_at']}"
        )
        with st.expander(header):
            col_old, col_new = st.columns(2)
            with col_old:
                st.markdown(f"**Bản CŨ** (v{c['old_version']}, status={c['old_status']})")
                st.caption(f"{c['old_source_file']} | {c['old_source_ref']}")
                st.text_area("old_content", c["old_content"], height=150,
                              key=f"old_{c['id']}", label_visibility="collapsed", disabled=True)
            with col_new:
                st.markdown(f"**Bản MỚI** (v{c['new_version']}, status={c['new_status']})")
                st.caption(f"{c['new_source_file']} | {c['new_source_ref']}")
                st.text_area("new_content", c["new_content"], height=150,
                              key=f"new_{c['id']}", label_visibility="collapsed", disabled=True)

            if c["conflict_status"] == "open":
                b1, b2, b3 = st.columns(3)
                if b1.button("✅ Dùng bản mới (supersede cũ)", key=f"use_new_{c['id']}"):
                    store.resolve_conflict(c["id"], "use_new")
                    st.rerun()
                if b2.button("↩️ Giữ bản cũ (supersede mới)", key=f"keep_old_{c['id']}"):
                    store.resolve_conflict(c["id"], "keep_old")
                    st.rerun()
                if b3.button("🙈 Bỏ qua (giữ cả 2)", key=f"ignore_{c['id']}"):
                    store.resolve_conflict(c["id"], "ignored")
                    st.rerun()
            else:
                st.caption(f"Đã xử lý lúc {c['resolved_at']}")
