# AWS Docs RAG

A hand-built Retrieval-Augmented Generation (RAG) pipeline that answers questions about AWS SageMaker and the Terraform AWS provider by retrieving relevant documentation and generating grounded answers — built entirely on AWS-native services (Bedrock, OpenSearch Serverless).

Built as a portfolio project to learn the mechanics of RAG from the ground up: chunking, embeddings, vector search, and retrieval-grounded generation — rather than relying on a managed framework.

## Architecture

```
GitHub doc repos (SageMaker + Terraform AWS provider, Markdown)
              │
              ▼
      Chunking (src/chunk.py)
   split by markdown headers, ~1500 char max per chunk
              │
              ▼
  Embedding (Bedrock Titan Embeddings v2)
              │
              ▼
   Vector index (OpenSearch Serverless)
              │
   ┌──────────┴──────────┐
   │   Query time:         │
   │   embed question       │
   │   → k-NN search         │
   │   → top-k chunks         │
   │   → Claude (Bedrock)      │
   │     generates grounded    │
   │     answer from context    │
   └────────────────────────────┘
```

## What's here

| Path | Purpose |
|---|---|
| `infrastructure/` | Terraform config — OpenSearch Serverless collection, IAM execution role, S3 staging bucket |
| `src/chunk.py` | Filters and chunks the source docs into retrieval-sized pieces |
| `src/index.py` | Embeds each chunk via Bedrock Titan and writes it to the OpenSearch vector index |
| `src/query.py` | Embeds a question, retrieves the top-k most relevant chunks, and asks Claude to answer grounded in that context |

## Corpus

Two documentation sources, cloned directly from their GitHub repos as Markdown (no HTML scraping):

- **SageMaker Developer Guide** ([awsdocs/amazon-sagemaker-developer-guide](https://github.com/awsdocs/amazon-sagemaker-developer-guide)) — filtered to ~209 pages covering training, deployment, endpoints, pipelines, and processing
- **Terraform AWS Provider docs** ([hashicorp/terraform-provider-aws](https://github.com/hashicorp/terraform-provider-aws)) — filtered to ~108 resource pages (`s3_*`, `iam_*`, `sagemaker_*`, `opensearchserverless_*`)

Chunked into 2,835 retrieval-sized pieces, split primarily on markdown headers to keep each chunk topically coherent.

> **Note on the SageMaker docs repo**: AWS archived and wiped their public docs repos in 2023 — the default branch now only contains a deprecation notice. The actual content is still intact on the `master` branch specifically, which is what this project clones from.

## Infrastructure

- **OpenSearch Serverless** collection (`VECTORSEARCH` type) — the vector index, with dedicated encryption, network, and data-access policies
- **IAM execution role** — scoped for Bedrock (`InvokeModel`) and OpenSearch Serverless (`aoss:APIAccessAll`) access
- **S3 bucket** — staging area for raw doc files, public access blocked, AES256 encryption

## Setup

**Prerequisites:** Python 3.12, an AWS account with Bedrock access, Terraform ≥ 1.5, AWS CLI configured with a named profile.

```bash
# 1. Provision infrastructure
cd infrastructure
terraform init
terraform apply

# 2. Set up Python environment
cd ..
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
pip install boto3 python-dotenv opensearch-py requests-aws4auth

# 3. Configure environment variables
cp .env.example .env
# edit .env — set OPENSEARCH_ENDPOINT from `terraform output opensearch_collection_endpoint` (strip the https://)

# 4. Clone the source docs
git clone --depth 1 --branch master https://github.com/awsdocs/amazon-sagemaker-developer-guide.git data/sagemaker-docs
git clone --depth 1 https://github.com/hashicorp/terraform-provider-aws.git data/terraform-provider-aws

# 5. Build the pipeline
python src/chunk.py
python src/index.py --limit 5   # test on a small slice first
python src/index.py             # full run (~2,835 chunks, 15-30 min)

# 6. Ask questions
python src/query.py "How do I deploy a SageMaker model to a real-time endpoint?"
```

## Cost note

**Unlike most portfolio infrastructure, OpenSearch Serverless bills continuously** — roughly $0.24/hour minimum for the 2 OCUs it requires, even fully idle. This is meaningfully different from typical pay-per-use AWS services (S3, Lambda, per-second SageMaker training jobs). Run `terraform destroy` in `infrastructure/` when not actively working on this project rather than leaving it running.

## What I learned

- OpenSearch Serverless vector-search collections don't support custom document IDs on indexing — IDs must be auto-assigned; custom IDs go in the document body as a regular field instead
- AWS archived their public docs-on-GitHub repos in 2023; content that appears in search results as "at master" can be misleading if the *default* branch was repointed elsewhere during the archive
- Bedrock model IDs are date-stamped snapshots that get retired — and newer/larger Claude models often require a cross-region inference profile prefix (`us.`) rather than direct on-demand invocation
- **A single retrieval technique isn't enough for every failure mode — and diagnosing *why* matters more than adding more techniques blindly.** I traced one specific query ("what arguments does `aws_iam_role` support?") through three progressively more sophisticated retrieval strategies:
  1. **Pure vector search** (top-3): retrieved the file's intro section, missed the actual Argument Reference section — it was 10 chunks away in an 18-chunk file, well outside typical top-k.
  2. **+ Neighbor-chunk retrieval** (±1 adjacent chunks): correctly expanded whatever was found into its complete surrounding context, but a 10-chunk gap is far outside a ±1 window — didn't close the gap.
  3. **+ Hybrid search** (BM25 keyword + vector similarity, fused via OpenSearch's normalization processor): improved retrieval quality broadly, but for this specific query, still centered on the intro chunk. Bumping `k` from 3 to 8 didn't help either — it surfaced six *other* IAM resources' "Argument Reference" sections, never the correct one.

  **Root cause, confirmed by direct diagnostic queries against the index**: `aws_iam_role` is a common token that appears throughout the corpus as a cross-reference (assumed-role principals, examples, related resources), and "Argument Reference" is generic boilerplate present in nearly every Terraform resource doc. Neither signal is discriminative enough on its own to disambiguate *this specific resource's* section from dozens of lexically-similar candidates elsewhere in the corpus.

  In every case, the model correctly said the context was insufficient rather than hallucinating an answer — across all three retrieval strategies, on this query, zero false answers.

  **Real fixes for this failure class** (not implemented, documented as next steps): metadata filtering (detect the resource name in the question, filter/boost chunks from the matching file before ranking), query rewriting (reformulate natural questions into phrasing closer to how docs are actually written), and reranking (retrieve a larger candidate set, then use a cross-encoder or LLM call to re-score specifically for relevance to the named entity).

## Roadmap

- [x] Infrastructure as code (Terraform — OpenSearch Serverless, IAM, S3)
- [x] Document chunking pipeline
- [x] Embedding + vector indexing (Bedrock Titan + OpenSearch)
- [x] Retrieval + grounded generation (Claude via Bedrock)
- [x] Validated against real queries — confirmed correct answers, correct corpus selection, and honest "insufficient context" responses rather than hallucination
- [x] Neighbor-chunk retrieval (pull adjacent chunks when one from a file scores highly)
- [x] Hybrid search (BM25 + vector, fused via OpenSearch Serverless search pipeline) — implemented and validated; genuinely improves retrieval broadly, but documented investigation shows it doesn't solve every gap (see "What I learned")
- [ ] Metadata filtering (detect resource/entity names in the question, filter or boost chunks from the matching source file)
- [ ] Query rewriting (reformulate natural-language questions into doc-like phrasing before retrieval)
- [ ] Reranking (retrieve a larger candidate set, re-score with a cross-encoder or LLM call for relevance to the named entity)
- [ ] Evaluation harness (measure retrieval relevance / answer quality systematically)
- [ ] Simple query interface (CLI polish or a minimal web UI)
- [ ] Incremental re-indexing (currently full-rebuild only)