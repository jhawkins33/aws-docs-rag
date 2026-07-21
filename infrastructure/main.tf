terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket  = "aws-docs-rag-tfstate"
    key     = "aws-docs-rag/terraform.tfstate"
    region  = "us-east-1"
    profile = "churn-mlops-personal"
  }
}

provider "aws" {
  region  = var.aws_region
  profile = "churn-mlops-personal"
}

data "aws_caller_identity" "current" {}