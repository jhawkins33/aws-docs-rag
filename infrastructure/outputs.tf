output "opensearch_collection_endpoint" {
  value = aws_opensearchserverless_collection.vector_store.collection_endpoint
}

output "rag_execution_role_arn" {
  value = aws_iam_role.rag_execution.arn
}

output "raw_docs_bucket" {
  value = aws_s3_bucket.raw_docs.bucket
}