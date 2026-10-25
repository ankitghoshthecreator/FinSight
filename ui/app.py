"""Streamlit interactive UI for FinSight RAG Enterprise Document Intelligence.
Role target: AI Engineer | JPMorgan Chase, Goldman Sachs, MUFG, UBS, BNY Mellon.
"""

import os
import json
import requests
import streamlit as st

st.set_page_config(
    page_title="FinSight RAG — Document Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar Configuration ───────────────────────────────────────────────────
st.sidebar.title("🏦 FinSight RAG")
st.sidebar.caption("Production-Grade Financial Document Intelligence")

api_url = st.sidebar.text_input("API Base URL", value="http://localhost:8000")
api_key = st.sidebar.text_input("X-API-Key (optional)", value="", type="password")

headers = {"Content-Type": "application/json"}
if api_key:
    headers["X-API-Key"] = api_key


# System Health & Stats in Sidebar
try:
    health_res = requests.get(f"{api_url}/health", headers=headers, timeout=3)
    if health_res.status_code == 200:
        h_data = health_res.json()
        st.sidebar.success("🟢 API Online")
        st.sidebar.metric("Indexed Chunks", h_data.get("indexed_chunks", 0))
        st.sidebar.metric("Unique Docs", h_data.get("unique_documents", 0))
        st.sidebar.caption(f"Backend: `{h_data.get('vector_backend')}` ({h_data.get('dense_encoder')})")
    else:
        st.sidebar.error("🔴 API Auth / Error")
except Exception:
    st.sidebar.warning("⚠️ API Offline (run `uvicorn api.main:app`)")


# Quick Load Sample Documents
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Pre-load Sample Filings")
sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "samples")

if os.path.exists(sample_dir):
    sample_files = [f for f in os.listdir(sample_dir) if f.endswith(".txt")]
    for s_file in sample_files:
        if st.sidebar.button(f"📥 Load {s_file[:20]}...", key=s_file):
            file_path = os.path.join(sample_dir, s_file)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            try:
                ingest_res = requests.post(
                    f"{api_url}/ingest/text",
                    headers=headers,
                    json={"filename": s_file, "content": content, "page_number": 1, "redact_pii": True},
                    timeout=10,
                )
                if ingest_res.status_code == 200:
                    st.sidebar.success(f"Indexed {s_file}!")
                    st.rerun()
                else:
                    st.sidebar.error(f"Failed: {ingest_res.text}")
            except Exception as e:
                st.sidebar.error(f"Error: {e}")


# ── Main Navigation Tabs ────────────────────────────────────────────────────
tab_query, tab_ingest, tab_docs, tab_metrics = st.tabs([
    "🔍 Financial QA & Query",
    "📄 Ingest Documents",
    "🗂️ Document Index",
    "📊 Observability & Metrics",
])


# ── TAB 1: Financial QA & Query ─────────────────────────────────────────────
with tab_query:
    st.subheader("Source-Grounded Financial Query")
    st.markdown("Answers are strictly verified against indexed documents with citation guardrails.")

    col_q1, col_q2 = st.columns([4, 1])
    with col_q1:
        query_input = st.text_input(
            "Enter financial or regulatory query:",
            placeholder="e.g. What is Goldman Sachs Supplementary Leverage Ratio SLR?",
        )
    with col_q2:
        top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=3)

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        enable_reranking = st.checkbox("ColBERT Late-Interaction Reranking", value=True)
    with col_opt2:
        use_stream = st.checkbox("Streaming SSE Response", value=False)

    if st.button("🚀 Run Pipeline", type="primary"):
        if not query_input.strip():
            st.warning("Please enter a query.")
        else:
            payload = {
                "query": query_input,
                "top_k": top_k,
                "enable_reranking": enable_reranking,
            }

            try:
                with st.spinner("Executing hybrid retrieval + ColBERT reranking + guardrails..."):
                    res = requests.post(f"{api_url}/query", headers=headers, json=payload, timeout=30)

                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "")
                    confidence = data.get("confidence", 0.0)
                    escalate = data.get("escalate", False)
                    sources = data.get("sources", [])

                    # Status Banner
                    if escalate:
                        st.error("⚠️ **Escalation Triggered:** Insufficient context or low confidence detected. Guardrail prevented hallucination.")
                    else:
                        st.success(f"✅ Grounded Response — Confidence: **{confidence * 100:.1f}%**")

                    st.markdown("### Answer")
                    st.write(answer)

                    # Citations
                    if sources:
                        st.markdown("### Citations & Source Evidence")
                        for idx, src in enumerate(sources, start=1):
                            with st.expander(f"📍 Citation {idx}: {src.get('doc_name')} (Page {src.get('page_number')})"):
                                st.markdown(f"**Excerpt:** *\"{src.get('snippet')}\"*")
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as exc:
                st.error(f"Request failed: {exc}")


# ── TAB 2: Ingest Documents ─────────────────────────────────────────────────
with tab_ingest:
    st.subheader("Ingest Financial Filings & Regulatory Documents")

    ingest_type = st.radio("Ingestion Method:", ["Upload File (PDF / DOCX / TXT)", "Paste Raw Text / Markdown"])

    if ingest_type == "Upload File (PDF / DOCX / TXT)":
        uploaded_file = st.file_uploader("Upload financial document", type=["pdf", "docx", "txt", "md"])
        async_mode = st.checkbox("Async Background Processing (for large 10-K filings)", value=False)

        if st.button("📥 Ingest Uploaded File") and uploaded_file:
            endpoint = "/ingest/file/async" if async_mode else "/ingest/file"
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            with st.spinner("Processing & chunking document..."):
                r = requests.post(f"{api_url}{endpoint}", headers={"X-API-Key": api_key} if api_key else {}, files=files)
            if r.status_code == 200:
                res_data = r.json()
                st.success("File ingested successfully!")
                st.json(res_data)
            else:
                st.error(f"Ingestion failed ({r.status_code}): {r.text}")

    else:
        doc_name = st.text_input("Document Name", value="Policy_Memo.txt")
        page_num = st.number_input("Page Number", min_value=1, value=1)
        redact = st.checkbox("Redact PII (SSN, PAN, Email, Bank Routing)", value=True)
        raw_text = st.text_area("Document Content", height=200, placeholder="Paste policy, filing, or KYC memo...")

        if st.button("📥 Ingest Text Document"):
            if raw_text.strip():
                p = {"filename": doc_name, "content": raw_text, "page_number": page_num, "redact_pii": redact}
                with st.spinner("Indexing into vector store..."):
                    r = requests.post(f"{api_url}/ingest/text", headers=headers, json=p)
                if r.status_code == 200:
                    st.success("Document text indexed successfully!")
                    st.json(r.json())
                else:
                    st.error(f"Failed ({r.status_code}): {r.text}")


# ── TAB 3: Document Index Management ────────────────────────────────────────
with tab_docs:
    st.subheader("Indexed Documents & Chunks")

    try:
        docs_res = requests.get(f"{api_url}/documents", headers=headers)
        if docs_res.status_code == 200:
            doc_data = docs_res.json()
            st.metric("Total Indexed Chunks", doc_data.get("total_chunks", 0))
            st.metric("Unique Document IDs", doc_data.get("unique_documents", 0))

            chunks = doc_data.get("chunks", [])
            if chunks:
                st.dataframe(chunks, use_container_width=True)
            else:
                st.info("No documents currently indexed in vector store.")
    except Exception as e:
        st.error(f"Failed to load document list: {e}")

    st.markdown("---")
    st.subheader("Danger Zone: Index Reset")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        del_doc_id = st.text_input("Delete specific Document ID", placeholder="doc_123456")
        if st.button("🗑️ Delete Document"):
            if del_doc_id:
                r = requests.delete(f"{api_url}/documents/{del_doc_id}", headers=headers)
                if r.status_code == 200:
                    st.success(f"Document {del_doc_id} deleted!")
                    st.rerun()
                else:
                    st.error(f"Delete failed: {r.text}")
    with col_d2:
        if st.button("⚠️ Reset Entire Index", type="secondary"):
            r = requests.delete(f"{api_url}/documents", headers=headers)
            if r.status_code == 200:
                st.warning("All chunks cleared!")
                st.rerun()


# ── TAB 4: Observability & Metrics ──────────────────────────────────────────
with tab_metrics:
    st.subheader("System Observability & Model Guardrail Metrics")

    try:
        m_res = requests.get(f"{api_url}/metrics", headers=headers)
        if m_res.status_code == 200:
            m = m_res.json()

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Total Requests", m.get("total_requests", 0))
            col_m2.metric("Total QA Queries", m.get("total_queries", 0))
            col_m3.metric("Escalation Rate", f"{m.get('escalation_rate_pct', 0.0)}%")
            col_m4.metric("Avg Latency", f"{m.get('avg_query_latency_ms', 0.0)} ms")

            col_m5, col_m6 = st.columns(2)
            with col_m5:
                st.markdown("#### Endpoint Traffic")
                st.bar_chart(m.get("endpoints", {}))
            with col_m6:
                st.markdown("#### HTTP Status Codes")
                st.bar_chart(m.get("status_codes", {}))

            with st.expander("Prometheus Exposition Format"):
                prom_res = requests.get(f"{api_url}/metrics?format=prometheus", headers=headers)
                st.code(prom_res.text if prom_res.status_code == 200 else "Unavailable", language="text")
    except Exception as e:
        st.error(f"Failed to fetch metrics: {e}")
