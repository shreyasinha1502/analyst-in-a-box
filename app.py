"""app.py — Streamlit UI for Analyst-in-a-Box.

Flow: upload dataset -> auto schema preview -> chat questions -> see generated
SQL + chart + written insight. Dataset-agnostic end to end.
"""

from __future__ import annotations

import os
import base64
import tempfile

import streamlit as st

# Allow API keys / config to come from Streamlit Secrets (never committed).
for _k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER",
           "ANTHROPIC_MODEL", "OPENAI_MODEL"):
    try:
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
    except Exception:
        pass

from agent import schema_reader, usage
from agent.orchestrator import answer_question

MAX_SESSION_QUESTIONS = int(os.environ.get("MAX_SESSION_QUESTIONS", "15"))

st.set_page_config(page_title="Analyst-in-a-Box", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state():
    ss = st.session_state
    ss.setdefault("db_path", None)
    ss.setdefault("schema", None)
    ss.setdefault("schema_text", None)
    ss.setdefault("chat", [])            # list of {"q":..., "result": AgentResult}
    ss.setdefault("history", [])         # provider-neutral msg history for follow-ups
    ss.setdefault("n_questions", 0)


_init_state()


# ---------------------------------------------------------------------------
# Sidebar: upload + provider
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 Analyst-in-a-Box")
    st.caption("Upload any tabular dataset and ask questions in plain English/Hindi.")

    provider = st.selectbox("LLM provider", ["anthropic", "openai"],
                            index=0 if os.environ.get("LLM_PROVIDER", "anthropic") == "anthropic" else 1)
    os.environ["LLM_PROVIDER"] = provider
    key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    if not os.environ.get(key_env):
        st.warning(f"{key_env} not set. Add it in Secrets / environment.", icon="🔑")

    uploaded = st.file_uploader("Dataset (CSV / XLSX)", type=["csv", "xlsx", "xls"])

    st.divider()
    st.caption("Or try a bundled sample:")
    sample = st.selectbox("Sample dataset", ["— none —", "sales", "sports", "education"])

    load_clicked = st.button("Load dataset", type="primary", use_container_width=True)

    st.divider()
    st.caption(f"Session: {st.session_state.n_questions}/{MAX_SESSION_QUESTIONS} questions · "
               f"Today: {usage.get_count()}/{usage.MAX_DAILY}")


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
def _load(file_obj_or_path, name: str, ftype: str | None):
    db_dir = tempfile.mkdtemp(prefix="aib_")
    db_path = os.path.join(db_dir, "dataset.db")
    schema_reader.load_flat_file_to_sqlite(file_obj_or_path, db_path,
                                           table_name=name, file_type=ftype)
    schema = schema_reader.introspect(db_path)
    st.session_state.db_path = db_path
    st.session_state.schema = schema
    st.session_state.schema_text = schema.to_prompt_text()
    st.session_state.chat = []
    st.session_state.history = []


if load_clicked:
    try:
        if uploaded is not None:
            ftype = uploaded.name.rsplit(".", 1)[-1].lower()
            _load(uploaded, uploaded.name.rsplit(".", 1)[0], ftype)
            st.success(f"Loaded {uploaded.name}")
        elif sample != "— none —":
            path = os.path.join("data", "sample_datasets", f"{sample}.csv")
            _load(path, sample, "csv")
            st.success(f"Loaded sample: {sample}")
        else:
            st.error("Upload a file or pick a sample first.")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")


# ---------------------------------------------------------------------------
# Schema preview
# ---------------------------------------------------------------------------
if st.session_state.schema is not None:
    schema = st.session_state.schema
    with st.expander("🔎 Auto-detected schema & preview", expanded=True):
        for t in schema.tables:
            st.markdown(f"**Table `{t.name}`** — {t.row_count} rows")
            import pandas as pd
            cols_df = pd.DataFrame([{
                "column": c.name, "type": c.dtype, "nulls": c.null_count,
                "distinct": c.distinct_count,
                "examples": ", ".join(str(v) for v in c.sample_values[:4]),
            } for c in t.columns])
            st.dataframe(cols_df, use_container_width=True, hide_index=True)
            if t.sample_rows:
                st.dataframe(pd.DataFrame(t.sample_rows), use_container_width=True, hide_index=True)
else:
    st.info("👈 Upload a dataset or load a sample to begin.")


# ---------------------------------------------------------------------------
# Chat history render
# ---------------------------------------------------------------------------
def _render_result(res):
    if res.sql_queries:
        with st.expander("🧾 Generated SQL", expanded=False):
            for q in res.sql_queries:
                st.code(q, language="sql")
    for ch in res.charts:
        img = base64.b64decode(ch["image_base64"])
        st.image(img, caption=ch.get("title"), use_container_width=True)
    if res.analyses:
        with st.expander("📈 Statistical analysis", expanded=False):
            for a in res.analyses:
                st.json(a)
    st.markdown(res.answer)
    if res.error:
        st.caption(f"⚠️ {res.error}")


for turn in st.session_state.chat:
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        _render_result(turn["result"])


# ---------------------------------------------------------------------------
# Chat input + agent call (with abuse caps)
# ---------------------------------------------------------------------------
question = st.chat_input("Ask about your data…" if st.session_state.schema else "Load a dataset first")

if question:
    if st.session_state.schema is None:
        st.warning("Load a dataset first.")
    elif st.session_state.n_questions >= MAX_SESSION_QUESTIONS:
        st.warning("Demo limit reached for this session — check the video walkthrough "
                   "or README for a full example run.", icon="🛑")
    elif usage.daily_limit_reached():
        st.warning("Daily demo limit reached — please try again tomorrow.", icon="🛑")
    else:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing…"):
                res = answer_question(
                    question,
                    db_path=st.session_state.db_path,
                    schema_text=st.session_state.schema_text,
                    provider=os.environ.get("LLM_PROVIDER"),
                    history=st.session_state.history,
                )
            _render_result(res)

        # bump counters + persist turn
        st.session_state.n_questions += 1
        usage.increment()
        st.session_state.chat.append({"q": question, "result": res})
        # keep a compact history so follow-ups have context
        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({"role": "assistant", "content": res.answer})
        # cap history length
        st.session_state.history = st.session_state.history[-12:]
