"""
Queries the RAG pipeline: embeds a question, retrieves the most
relevant chunks from OpenSearch, and asks Bedrock Claude to answer
using only that retrieved context.

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


def retrieve(opensearch, bedrock, question, k=TOP_K):
    query_embedding = embed_text(bedrock, question)
    response = opensearch.search(
        index=INDEX_NAME,
        body={
            "size": k,
            "query": {"knn": {"embedding": {"vector": query_embedding, "k": k}}},
        },
    )
    return [hit["_source"] for hit in response["hits"]["hits"]]


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

    print(f"Retrieving top {TOP_K} chunks for: {args.question}\n")
    chunks = retrieve(opensearch, bedrock, args.question)

    for i, c in enumerate(chunks):
        print(f"--- Chunk {i+1}: {c['source']}/{c['file']} ---")
        print(c["text"][:200] + "...\n")

    print("Generating answer...\n")
    answer = generate_answer(bedrock, args.question, chunks)
    print(f"ANSWER:\n{answer}")


if __name__ == "__main__":
    main()