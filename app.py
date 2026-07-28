"""
Streamlit UI for the AWS Docs RAG pipeline.

Usage:
    streamlit run app.py
"""

import streamlit as st
from src.query import (
    get_clients,
    ensure_search_pipeline,
    hybrid_retrieve,
    fetch_neighbors,
    generate_answer,
)

st.set_page_config(page_title="AWS Docs RAG", page_icon="📚", layout="centered")


@st.cache_resource
def load_clients():
    """Cache the AWS clients and search pipeline setup across reruns."""
    bedrock, opensearch = get_clients()
    ensure_search_pipeline(opensearch)
    return bedrock, opensearch


def expand_with_neighbors(opensearch, top_hits):
    seen = {(c["file"], c["chunk_index"]) for c in top_hits}
    chunks = list(top_hits)
    for chunk in top_hits:
        for neighbor in fetch_neighbors(opensearch, chunk):
            key = (neighbor["file"], neighbor["chunk_index"])
            if key not in seen:
                seen.add(key)
                chunks.append(neighbor)
    return chunks


st.title("📚 AWS Docs RAG")
st.caption(
    "Ask questions about SageMaker and the Terraform AWS provider. "
    "Answers are grounded in retrieved documentation — hybrid search "
    "(BM25 + vector) with metadata filtering for Terraform resources."
)

bedrock, opensearch = load_clients()

question = st.text_input(
    "Your question",
    placeholder="e.g. What arguments does aws_iam_role support?",
)

if st.button("Ask", type="primary") and question:
    with st.spinner("Retrieving and generating..."):
        top_hits = hybrid_retrieve(opensearch, bedrock, question, k=8)
        chunks = expand_with_neighbors(opensearch, top_hits)
        answer = generate_answer(bedrock, question, chunks)

    st.markdown("### Answer")
    st.markdown(answer)

    with st.expander(f"Retrieved {len(chunks)} source chunks"):
        for i, c in enumerate(chunks):
            st.markdown(f"**{i+1}. {c['source']}/{c['file']}** (chunk {c['chunk_index']})")
            st.text(c["text"][:300] + ("..." if len(c["text"]) > 300 else ""))
            st.divider()