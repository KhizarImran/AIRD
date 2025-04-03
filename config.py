"""
Configuration settings for the Corrosion Detection System.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "corrosion-detection-images")
MODEL_S3_PATH = os.getenv("MODEL_S3_PATH", "models/corrosion_detector")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "anthropic.claude-3-sonnet-20240229")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))  # Threshold for positive corrosion detection

# Application Configuration
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", str(16 * 1024 * 1024)))  # 16MB max file size

# SageMaker Configuration (only used if deploying through SageMaker)
SAGEMAKER_ROLE = os.getenv("SAGEMAKER_ROLE", "SageMakerExecutionRole")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE", "ml.g4dn.xlarge")  # GPU instance for training/inference

# API Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
USE_AWS_BEDROCK = os.getenv("USE_AWS_BEDROCK", "True").lower() == "true"  # Default to using AWS Bedrock 