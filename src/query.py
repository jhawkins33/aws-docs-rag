"""
Queries the RAG pipeline: embeds a question, retrieves the most
relevant chunks from OpenSearch (plus their immediate neighbors from
the same source file), and asks Bedrock Claude to answer using only
that retrieved context.

Usage:
    python src/query.py "How do I deploy a SageMaker endpoint?"
"""

import argparse
import re
import json
import os
import boto3
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")
PROFILE = os.environ.get("AWS_PROFILE", "churn-mlops-personal")
OPENSEARCH_ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"]
INDEX_NAME = "docs-index"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
GENERATION_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
TOP_K = 3


def get_clients():
    session = boto3.Session(profile_name=PROFILE, region_name=REGION)
    bedrock = session.client("bedrock-runtime")

    credentials = session.get_credentials()
    auth = AWSV4SignerAuth(credentials, REGION, "aoss")

    opensearch = OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
    )
    return bedrock, opensearch


def embed_text(bedrock, text):
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps({"inputText": text}),
    )
    result = json.loads(response["body"].read())
    return result["embedding"]
    
def rewrite_query(bedrock, question):
    """
    Reformulate the user's natural-language question into phrasing
    closer to how technical documentation is written — bridging the
    vocabulary gap between how people ask and how docs are worded.

    Example:
        "What arguments does aws_iam_role support?"
        -> "aws_iam_role resource arguments required optional Terraform"
    """
    prompt = f"""Rewrite the following question into a concise search query optimized for retrieving relevant technical documentation. Use terminology and phrasing that would appear in the documentation itself rather than conversational language. Output only the rewritten query, nothing else.

Question: {question}

Rewritten query:"""

    response = bedrock.invoke_model(
        modelId=GENERATION_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"].strip()


def fetch_neighbors(opensearch, chunk):
    """Fetch the chunk immediately before and after this one in the same file."""
    neighbors = []
    for offset in (-1, 1):
        neighbor_idx = chunk["chunk_index"] + offset
        if neighbor_idx < 0 or neighbor_idx >= chunk["total_chunks_in_file"]:
            continue
        response = opensearch.search(
            index=INDEX_NAME,
            body={
                "size": 1,
                "query": {
                    "bool": {
                        "filter": [
                            {"term": {"file": chunk["file"]}},
                            {"term": {"chunk_index": neighbor_idx}},
                        ]
                    }
                },
            },
        )
        hits = response["hits"]["hits"]
        if hits:
            neighbors.append(hits[0]["_source"])
    return neighbors

SEARCH_PIPELINE_NAME = "hybrid-search-pipeline"


def ensure_search_pipeline(opensearch):
    """
    Create the search pipeline that normalizes and combines BM25
    (keyword) and k-NN (vector) scores for hybrid search. Idempotent —
    safe to call every run.
    """
    body = {
        "description": "Normalize and combine BM25 + kNN scores for hybrid search",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [0.3, 0.7]},
                    },
                }
            }
        ],
    }
    opensearch.transport.perform_request(
        "PUT", f"/_search/pipeline/{SEARCH_PIPELINE_NAME}", body=body
    )


def extract_file_filter(question):
    """
    Detect a Terraform resource name (e.g. aws_iam_role) mentioned in
    the question and map it to its corresponding doc filename, following
    the Terraform provider docs' naming convention: aws_iam_role ->
    iam_role.html.markdown. Scoped to Terraform docs only — SageMaker
    doc filenames don't follow a predictable pattern from question text.
    """
    match = re.search(r"\baws_[a-z0-9_]+\b", question)
    if not match:
        return None
    resource_name = match.group(0)
    return resource_name[len("aws_"):] + ".html.markdown"


def hybrid_retrieve(opensearch, bedrock, question, k=TOP_K):
    """
    Retrieve using hybrid search: BM25 keyword match + k-NN vector
    similarity, combined via the search pipeline's weighted average.
    If the question mentions a specific Terraform resource (e.g.
    aws_iam_role), filter to only that resource's doc file first —
    eliminates competition from other, lexically-similar resources.
    """
    query_embedding = embed_text(bedrock, question)

    file_filter = extract_file_filter(question)

    match_query = {"match": {"text": {"query": question}}}
    knn_query = {"knn": {"embedding": {"vector": query_embedding, "k": k}}}

    if file_filter:
        print(f"(filtering to file: {file_filter})")
        match_query = {
            "bool": {
                "must": [match_query],
                "filter": [{"term": {"file": file_filter}}],
            }
        }
        knn_query["knn"]["embedding"]["filter"] = {"term": {"file": file_filter}}

    hybrid_query = {"queries": [match_query, knn_query]}

    body = {"size": k, "query": {"hybrid": hybrid_query}}

    response = opensearch.search(
        index=INDEX_NAME,
        body=body,
        params={"search_pipeline": SEARCH_PIPELINE_NAME},
    )
    hits = [hit["_source"] for hit in response["hits"]["hits"]]

    # If a confident filter produced zero results (e.g. the detected
    # resource doesn't actually exist in this corpus), fall back to
    # unfiltered hybrid search rather than returning nothing.
    if file_filter and not hits:
        print("(filter produced no results, falling back to unfiltered search)")
        hybrid_query.pop("filter", None)
        body = {"size": k, "query": {"hybrid": hybrid_query}}
        response = opensearch.search(
            index=INDEX_NAME,
            body=body,
            params={"search_pipeline": SEARCH_PIPELINE_NAME},
        )
        hits = [hit["_source"] for hit in response["hits"]["hits"]]

    return hits


def retrieve(opensearch, bedrock, question, k=TOP_K):
    query_embedding = embed_text(bedrock, question)
    response = opensearch.search(
        index=INDEX_NAME,
        body={
            "size": k,
            "query": {"knn": {"embedding": {"vector": query_embedding, "k": k}}},
        },
    )
    top_hits = [hit["_source"] for hit in response["hits"]["hits"]]

    seen = {(c["file"], c["chunk_index"]) for c in top_hits}
    expanded = list(top_hits)
    for chunk in top_hits:
        for neighbor in fetch_neighbors(opensearch, chunk):
            key = (neighbor["file"], neighbor["chunk_index"])
            if key not in seen:
                seen.add(key)
                expanded.append(neighbor)

    return expanded


def generate_answer(bedrock, question, chunks):
    context = "\n\n---\n\n".join(
        f"[{c['source']}/{c['file']}]\n{c['text']}" for c in chunks
    )
    prompt = f"""Answer the question using ONLY the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""

    response = bedrock.invoke_model(
        modelId=GENERATION_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]
    
def rerank_chunks(bedrock, question, chunks, threshold=3):
    """
    Score each retrieved chunk for relevance to the question using
    Claude, then sort by score and filter out chunks below the
    threshold. This gives generation a cleaner, more focused context.

    Scoring scale: 1 (not relevant) to 5 (highly relevant).
    Default threshold: 3 (keep chunks that are at least somewhat relevant).
    """
    if not chunks:
        return chunks

    scored = []
    for chunk in chunks:
        prompt = f"""Score the relevance of the following document chunk to the question on a scale of 1-5.

1 = Not relevant at all
2 = Slightly relevant
3 = Somewhat relevant
4 = Relevant
5 = Highly relevant

Question: {question}

Document chunk (from {chunk['source']}/{chunk['file']}):
{chunk['text'][:500]}

Respond with ONLY a single integer (1, 2, 3, 4, or 5), nothing else."""

        response = bedrock.invoke_model(
            modelId=GENERATION_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 5,
                "messages": [{"role": "user", "content": prompt}],
            }),
        )
        result = json.loads(response["body"].read())
        raw = result["content"][0]["text"].strip()
        try:
            score = int(raw[0])
        except (ValueError, IndexError):
            score = 3  # default to neutral if unparseable

        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [chunk for score, chunk in scored if score >= threshold]

    print(f"  Reranking: {len(chunks)} chunks → {len(reranked)} kept (threshold={threshold})")
    return reranked if reranked else [scored[0][1]]  # always keep at least 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()

    bedrock, opensearch = get_clients()
    ensure_search_pipeline(opensearch)

    search_query = rewrite_query(bedrock, args.question)
    print(f"Original:  {args.question}")
    print(f"Rewritten: {search_query}\n")
    top_hits = hybrid_retrieve(opensearch, bedrock, search_query, k=8)

    # Expand with neighbors, same logic as pure-vector retrieve()
    seen = {(c["file"], c["chunk_index"]) for c in top_hits}
    chunks = list(top_hits)
    for chunk in top_hits:
        for neighbor in fetch_neighbors(opensearch, chunk):
            key = (neighbor["file"], neighbor["chunk_index"])
            if key not in seen:
                seen.add(key)
                chunks.append(neighbor)
    print(f"Retrieved {len(chunks)} chunks total.\n")

    for i, c in enumerate(chunks):
        print(f"--- Chunk {i+1}: {c['source']}/{c['file']} (index {c['chunk_index']}) ---")
        print(c["text"][:200] + "...\n")

    # Only rerank when there are enough chunks that filtering adds value.
    # With fewer than 8 chunks, the context is already focused enough
    # that reranking risks over-filtering marginally-relevant content.
    if len(chunks) >= 8:
        chunks = rerank_chunks(bedrock, args.question, chunks, threshold=2)
    else:
        print(f"  Skipping rerank ({len(chunks)} chunks — below threshold)")
    print("Generating answer...\n")
    answer = generate_answer(bedrock, args.question, chunks)
    print(f"ANSWER:\n{answer}")


if __name__ == "__main__":
    main()