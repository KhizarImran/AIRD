# Corrosion Detection System

This project implements a multimodal AI system for detecting corrosion in industrial pipe images captured by drones, using Claude 3 Sonnet via AWS Bedrock for image analysis.

## Overview

This system allows you to upload images of industrial pipes and get an assessment of whether they contain corrosion. The system leverages Claude 3 Sonnet's multimodal capabilities through AWS Bedrock to analyze the images with high accuracy.

## Setup

1. Clone this repository in your SageMaker Studio or Notebook instance
2. Copy `.env.example` to `.env` and add your AWS credentials
3. Navigate to the project directory
4. Run `pip install -r requirements.txt`
5. Launch the application with `python app.py`

## AWS Credentials Configuration

You need to provide AWS credentials to access Bedrock. Two options are available:

1. **Explicit Credentials**: Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in your `.env` file
2. **IAM Role**: If running on an AWS service with an IAM role that has Bedrock access, no explicit credentials needed

Note: Your AWS role/user must have permissions to access the Claude 3 Sonnet model in AWS Bedrock.

## Usage

1. Upload one or more images through the web interface
2. The system will process the images using Claude 3 Sonnet via AWS Bedrock
3. Results will display which images contain corrosion with confidence scores
4. Each result includes detailed analysis from the AI explaining the reasoning

## Configuration

The system is configured by environment variables and can be customized in your `.env` file:

```
# AWS Settings
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
USE_AWS_BEDROCK=True

# Model Settings
CONFIDENCE_THRESHOLD=0.7
```

You can also change the Claude model by updating the MODEL_NAME variable in your `.env` file:

```
MODEL_NAME=anthropic.claude-3-sonnet-20240229
```

## Components

- `/src` - Core Python backend code, including AWS Bedrock integration
- `/app` - Web application for image upload and result visualization
- `/uploads` - Temporary storage for uploaded images

## How It Works

This system uses a functional programming approach to:

1. Accept image uploads via a Flask web interface
2. Send those images to Claude 3 Sonnet via AWS Bedrock with a specialized prompt for corrosion detection
3. Parse the AI's analysis into structured data
4. Visualize the results with confidence scores and detailed explanations

## Requirements

- Python 3.8+
- AWS account with Bedrock access to Claude 3 Sonnet
- Required Python packages listed in requirements.txt 