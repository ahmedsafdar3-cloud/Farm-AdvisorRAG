"""
Streamlit demo UI for Farm Advisor RAG.

Run: streamlit run src/app.py
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from rag_engine import RAGEngine

st.set_page_config(page_title="Farm Advisor", page_icon="🌿", layout="centered")

st.title("🌿 Farm Advisor")
st.caption(
    "RAG-powered agricultural advisory assistant — retrieval-augmented answers "
    "grounded in an agronomy knowledge base, with cited sources."
)

if not os.environ.get("GROQ_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
    st.warning(
        "No API key set. The app will still retrieve relevant sources, "
        "but won't generate a natural-language answer. Get a free key at "
        "console.groq.com/keys and add it to your .env file to enable full answers.",
        icon="⚠️",
    )


@st.cache_resource
def load_engine():
    return RAGEngine(top_k=4)


try:
    engine = load_engine()
except FileNotFoundError as e:
    st.error(f"{e}\n\nRun `python src/ingest.py` first to build the knowledge base index.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source']}** — {s['section']}")

if query := st.chat_input("Ask about crop diseases, soil, irrigation, harvest timing..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            result = engine.answer(query)
        st.markdown(result["answer"])
        with st.expander("Sources"):
            for s in result["sources"]:
                st.markdown(f"- **{s['source']}** — {s['section']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

with st.sidebar:
    st.subheader("About this project")
    st.markdown(
        """
This is a **hybrid RAG** system:
- **Dense retrieval** via sentence-transformer embeddings (semantic/paraphrase matching)
- **Sparse retrieval** via BM25 (exact keyword/jargon matching)
- Fused with **Reciprocal Rank Fusion**
- Answers are grounded strictly in retrieved context, with inline citations

Knowledge base: 5 agronomy documents covering olive diseases/pests, soil health,
irrigation, harvest planning, and field crop pest management.
        """
    )
    st.markdown("---")
    st.caption("Try: *'My olive leaves are yellowing and it's been raining a lot, what should I check?'*")
