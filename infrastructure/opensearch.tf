# Encryption policy — required before collection creation
resource "aws_opensearchserverless_security_policy" "encryption" {
  name = "${var.project}-encryption"
  type = "encryption"
  policy = jsonencode({
    Rules = [
      {
        ResourceType = "collection"
        Resource     = ["collection/${var.project}-*"]
      }
    ]
    AWSOwnedKey = true
  })
}

# Network policy — controls access to the collection endpoint
resource "aws_opensearchserverless_security_policy" "network" {
  name = "${var.project}-network"
  type = "network"
  policy = jsonencode([
    {
      Rules = [
        {
          ResourceType = "collection"
          Resource     = ["collection/${var.project}-*"]
        }
      ]
      AllowFromPublic = true
    }
  ])
}

# Data access policy — who/what can read and write vectors
resource "aws_opensearchserverless_access_policy" "data_access" {
  name = "${var.project}-data-access"
  type = "data"
  policy = jsonencode([
    {
      Rules = [
        {
          ResourceType = "collection"
          Resource     = ["collection/${var.project}-*"]
          Permission   = ["aoss:*"]
        },
        {
          ResourceType = "index"
          Resource     = ["index/${var.project}-*/*"]
          Permission   = ["aoss:*"]
        }
      ]
      Principal = [
        aws_iam_role.rag_execution.arn,
        data.aws_caller_identity.current.arn
      ]
    }
  ])
}

resource "aws_opensearchserverless_collection" "vector_store" {
  name = "${var.project}-vectors"
  type = "VECTORSEARCH"

  depends_on = [
    aws_opensearchserverless_security_policy.encryption,
    aws_opensearchserverless_security_policy.network
  ]
}