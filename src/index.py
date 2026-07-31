"""
Embeds chunks via Bedrock Titan and indexes them into OpenSearch Serverless.

Usage:
    python src/index.py --limit 20   # test on a small slice first
    python src/index.py              # full run
"""

import argparse
import json
import os
import time
import boto3
from dotenv import load_dotenv
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

load_dotenv()

REGION = os.environ.get("AWS_REGION", "us-east-1")
PROFILE = os.environ.get("AWS_PROFILE", "churn-mlops-personal")
OPENSEARCH_ENDPOINT = os.environ["OPENSEARCH_ENDPOINT"]
INDEX_NAME = "docs-index"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"
EMBEDDING_DIM = 1024

CHUNKS_PATH = "data/chunks.jsonl"


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
        timeout=60,
    )
    return bedrock, opensearch


def embed_text(bedrock, text):
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps({"inputText": text}),
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def ensure_index(opensearch):
    if opensearch.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' already exists.")
        return

    print(f"Creating index '{INDEX_NAME}'...")
    opensearch.indices.create(
        index=INDEX_NAME,
        body={
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": EMBEDDING_DIM,
                        "method": {
                            "name": "hnsw",
                            "engine": "faiss",
                            "space_type": "cosinesimil",
                        },
                    },
                    "text": {"type": "text"},
                    "source": {"type": "keyword"},
                    "file": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "total_chunks_in_file": {"type": "integer"},
                    "content_hash": {"type": "keyword"},
                }
            },
        },
    )
    print("Index created.")

def get_indexed_hashes(opensearch):
    """
    Retrieve all content_hash values currently in the index using
    search_after pagination (scroll is not supported by OpenSearch
    Serverless). Returns a set of hashes to skip during incremental
    indexing.
    """
    hashes = set()
    try:
        page_size = 1000
        last_sort = None

        while True:
            body = {
                "size": page_size,
                "_source": ["content_hash"],
                "query": {"match_all": {}},
                "sort": [{"_id": "asc"}],
            }
            if last_sort:
                body["search_after"] = last_sort

            response = opensearch.search(index=INDEX_NAME, body=body)
            hits = response["hits"]["hits"]
            if not hits:
                break

            for hit in hits:
                h = hit.get("_source", {}).get("content_hash")
                if h:
                    hashes.add(h)

            if len(hits) < page_size:
                break
            last_sort = hits[-1]["sort"]

    except Exception as e:
        print(f"  Warning: could not retrieve indexed hashes ({e}). Falling back to full index.")
        return set()

    return hashes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N chunks (for testing)")
    parser.add_argument("--full-rebuild", action="store_true", help="Re-index all chunks, ignoring existing index content")
    args = parser.parse_args()

    bedrock, opensearch = get_clients()
    ensure_index(opensearch)

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    if args.limit:
        chunks = chunks[: args.limit]

    if not args.full_rebuild:
        indexed_hashes = get_indexed_hashes(opensearch)
        chunks = [c for c in chunks if c.get("content_hash") not in indexed_hashes]
        print(f"Incremental mode: {len(chunks)} new/changed chunks to index (skipping already-indexed).")
    else:
        print(f"Full rebuild mode: indexing all {len(chunks)} chunks.")

    print(f"Indexing {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        if not chunks:
            print("Nothing to index.")
            return
        embedding = embed_text(bedrock, chunk["text"])
        opensearch.index(
            index=INDEX_NAME,
            body={
                "chunk_id": chunk["id"],
                "embedding": embedding,
                "text": chunk["text"],
                "source": chunk["source"],
                "file": chunk["file"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks_in_file": chunk["total_chunks_in_file"],
                "content_hash": chunk.get("content_hash", ""),
            },
        )
        if (i + 1) % 25 == 0 or (i + 1) == len(chunks):
            print(f"  {i + 1}/{len(chunks)} indexed")
        time.sleep(0.05)

    print("Done.")


if __name__ == "__main__":
    main()