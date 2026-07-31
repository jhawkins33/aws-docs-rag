"""
Chunks the filtered SageMaker + Terraform markdown docs into
retrieval-sized pieces, split primarily on markdown headers so each
chunk stays topically coherent, with a fallback character-based split
for any section that's still too long.

Usage:
    python src/chunk.py
"""

import hashlib
import json
import re
from pathlib import Path

SAGEMAKER_DIR = Path("data/sagemaker-docs/doc_source")
TERRAFORM_DIR = Path("data/terraform-provider-aws/website/docs/r")

SAGEMAKER_KEYWORDS = ["sagemaker-projects", "train", "deploy", "endpoint", "pipeline", "processing"]
TERRAFORM_PREFIXES = ("s3_", "iam_", "sagemaker_", "opensearchserverless_")

MAX_CHUNK_CHARS = 1500
OVERLAP_CHARS = 150

OUTPUT_PATH = Path("data/chunks.jsonl")


def get_sagemaker_files():
    pattern = re.compile("|".join(SAGEMAKER_KEYWORDS))
    return [f for f in SAGEMAKER_DIR.glob("*.md") if pattern.search(f.name)]


def get_terraform_files():
    return [f for f in TERRAFORM_DIR.glob("*.html.markdown") if f.name.startswith(TERRAFORM_PREFIXES)]


def split_by_headers(text: str):
    """Split markdown on ## / ### headers, keeping the header with its section."""
    parts = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
    return [p.strip() for p in parts if p.strip()]

def split_long_section(text: str):
    """Fallback: break an oversized section into overlapping windows."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + MAX_CHUNK_CHARS
        chunks.append(text[start:end])
        start = end - OVERLAP_CHARS
    return chunks


def chunk_file(path: Path, source: str):
    text = path.read_text(encoding="utf-8", errors="ignore")
    sections = split_by_headers(text)
    pieces = []
    for section in sections:
        pieces.extend(split_long_section(section))

    chunks = []
    for idx, piece in enumerate(pieces):
        content_hash = hashlib.sha256(piece.encode("utf-8")).hexdigest()[:16]
        chunks.append({
            "source": source,
            "file": path.name,
            "text": piece,
            "chunk_index": idx,
            "total_chunks_in_file": len(pieces),
            "content_hash": content_hash,
        })
    return chunks


def main():
    all_chunks = []

    sagemaker_files = get_sagemaker_files()
    print(f"Found {len(sagemaker_files)} SageMaker doc files")
    for f in sagemaker_files:
        all_chunks.extend(chunk_file(f, "sagemaker-docs"))

    terraform_files = get_terraform_files()
    print(f"Found {len(terraform_files)} Terraform doc files")
    for f in terraform_files:
        all_chunks.extend(chunk_file(f, "terraform-aws-provider"))

    print(f"Total chunks: {len(all_chunks)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for i, chunk in enumerate(all_chunks):
            chunk["id"] = f"chunk-{i}"
            out.write(json.dumps(chunk) + "\n")

    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()