"""Streamlit chat app.  Run:  streamlit run app.py"""
import streamlit as st

from config import SCHEMES
from rag_pipeline import ask

st.set_page_config(page_title="Yojana Saathi", page_icon="🌾")
st.title("🌾 Yojana Saathi")
st.caption("Ask about government schemes in English or Hindi. "
           "Answers come only from official documents.")

# ---------------- sidebar
with st.sidebar:
    scheme = st.selectbox("Choose a scheme", list(SCHEMES.values()))
    show_debug = st.toggle("Show debug info", value=False)
    if st.button("Clear chat"):
        st.session_state.history = []
        st.session_state.display = []
        st.rerun()
    st.markdown("---")
    st.markdown("Please don't share Aadhaar, bank or phone numbers. "
                "If you do, they are hidden automatically.")

# ---------------- state
if "history" not in st.session_state:
    st.session_state.history = []      # used by the pipeline (memory)
if "display" not in st.session_state:
    st.session_state.display = []      # what we show on screen (with sources)

# switching scheme starts a new chat so answers don't mix
if st.session_state.get("scheme") != scheme:
    st.session_state.scheme = scheme
    st.session_state.history = []
    st.session_state.display = []


def show_message(msg):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for file_name, page in msg["sources"]:
                    st.write(f"📄 {file_name} — page {page}")
        if show_debug and msg.get("debug"):
            st.json(msg["debug"])


for msg in st.session_state.display:
    show_message(msg)

# ---------------- chat input
question = st.chat_input("e.g. Who is eligible for PM-KISAN?")
if question:
    show_message({"role": "user", "content": question})
    with st.spinner("Searching official documents..."):
        try:
            r = ask(question, scheme, st.session_state.history)
        except Exception as e:  # network / API key problems
            r = {"answer": f"Sorry, something went wrong: {e}", "sources": []}

    if r.get("pii_masked"):
        st.info(f"Hidden for your privacy: {', '.join(r['pii_masked'])}")

    user_msg = {"role": "user", "content": question}
    bot_msg = {
        "role": "assistant",
        "content": r["answer"],
        "sources": r.get("sources", []),
        "debug": {k: r.get(k) for k in
                  ("rewritten", "blocked_reason", "grounded", "latency_s")},
    }
    st.session_state.display += [user_msg, bot_msg]
    show_message(bot_msg)
