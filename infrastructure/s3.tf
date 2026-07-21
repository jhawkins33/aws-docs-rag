resource "aws_s3_bucket" "raw_docs" {
  bucket = "${var.project}-raw-docs-${var.environment}"
}

resource "aws_s3_bucket_public_access_block" "raw_docs" {
  bucket                  = aws_s3_bucket.raw_docs.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "raw_docs" {
  bucket = aws_s3_bucket.raw_docs.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}