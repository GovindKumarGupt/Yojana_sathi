"""Retriever checker. Run:  streamlit run retriever_app.py

Type a question and see which chunks the retriever finds, with score, file and page.
No answer is written here, so you can check ONLY the search part.
"""
import streamlit as st

from config import SCHEMES, TOP_K, MIN_RELEVANCE
from memory import rewrite_question
from retriever import retrieve

st.set_page_config(page_title="Retriever checker", page_icon="🔎", layout="wide")
st.title("🔎 Retriever checker")
st.caption("Check if the search returns the right chunks. No answer is generated here.")

# ---------------- sidebar settings
with st.sidebar:
    scheme = st.selectbox("Scheme", list(SCHEMES.values()))
    top_k = st.slider("How many chunks (top_k)", 1, 10, TOP_K)
    use_rewrite = st.toggle(
        "Rewrite question first (like the real chatbot)", value=True,
        help="Translates Hindi to English. Needed for Hindi questions.")
    st.markdown(f"Chunks with score below **{MIN_RELEVANCE}** are treated as "
                "'not relevant' by the chatbot.")

question = st.text_input("Your question", placeholder="e.g. Who is not eligible?")

if question:
    with st.spinner("Searching..."):
        # step 1 (optional): same rewrite as the real pipeline
        search_q = rewrite_question(question, [], scheme) if use_rewrite else question
        # step 2: the actual search
        chunks = retrieve(search_q, scheme, top_k=top_k)

    st.write(f"**Question used for search:** {search_q}")

    if not any(c["score"] >= MIN_RELEVANCE for c in chunks):
        st.warning("No chunk reaches the relevance threshold, "
                   "so the chatbot would say 'I don't know'.")

    # ---------------- show each chunk
    for rank, c in enumerate(chunks, start=1):
        ok = c["score"] >= MIN_RELEVANCE
        label = f"#{rank}  |  {c['file_name']}, page {c['page']}  |  score {c['score']}"
        with st.expander(("✅ " if ok else "⚠️ ") + label, expanded=(rank <= 2)):
            st.progress(max(0.0, min(1.0, c["score"])))   # bar = how similar (0..1)
            st.write(c["text"])
