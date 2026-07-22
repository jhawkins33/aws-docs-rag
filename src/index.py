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
                }
            },
        },
    )
    print("Index created.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N chunks (for testing)")
    args = parser.parse_args()

    bedrock, opensearch = get_clients()
    ensure_index(opensearch)

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    if args.limit:
        chunks = chunks[: args.limit]

    print(f"Indexing {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
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
            },
        )
        if (i + 1) % 25 == 0 or (i + 1) == len(chunks):
            print(f"  {i + 1}/{len(chunks)} indexed")
        time.sleep(0.05)

    print("Done.")


if __name__ == "__main__":
    main()