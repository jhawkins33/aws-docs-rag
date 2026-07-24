"""
Queries the RAG pipeline: embeds a question, retrieves the most
relevant chunks from OpenSearch (plus their immediate neighbors from
the same source file), and asks Bedrock Claude to answer using only
that retrieved context.

Usage:
    python src/query.py "How do I deploy a SageMaker endpoint?"
"""

import argparse
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


def hybrid_retrieve(opensearch, bedrock, question, k=TOP_K):
    """
    Retrieve using hybrid search: BM25 keyword match + k-NN vector
    similarity, combined via the search pipeline's weighted average
    (30% keyword, 70% vector — vector still leads, but exact term
    matches like "Argument Reference" now get real weight too).
    """
    query_embedding = embed_text(bedrock, question)

    body = {
        "size": k,
        "query": {
            "hybrid": {
                "queries": [
                    {"match": {"text": {"query": question}}},
                    {"knn": {"embedding": {"vector": query_embedding, "k": k}}},
                ]
            }
        },
    }

    response = opensearch.search(
        index=INDEX_NAME,
        body=body,
        params={"search_pipeline": SEARCH_PIPELINE_NAME},
    )
    
    return [hit["_source"] for hit in response["hits"]["hits"]]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()

    bedrock, opensearch = get_clients()

    print(f"Retrieving top {TOP_K} chunks (plus neighbors) for: {args.question}\n")
    ensure_search_pipeline(opensearch)
    top_hits = hybrid_retrieve(opensearch, bedrock, args.question, k=8)

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

    print("Generating answer...\n")
    answer = generate_answer(bedrock, args.question, chunks)
    print(f"ANSWER:\n{answer}")


if __name__ == "__main__":
    main()