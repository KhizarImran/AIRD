"""
Corrosion Detection System - SageMaker Deployment Demo

This script demonstrates how to deploy and use the corrosion detection model in Amazon SageMaker.

The script covers:
1. Setting up the environment
2. Loading the model
3. Deploying the model to a SageMaker endpoint
4. Testing the model with sample images
5. Visualizing the results
"""

import os
import sys
import json
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import base64
import io

# Add the project root to the Python path
sys.path.append('..')

# Import project modules
import config
from src.detector import CorrosionDetector
from src.utils.image_utils import visualize_results, create_heatmap

# Set AWS region
region = config.AWS_REGION
print(f"AWS Region: {region}")

# Create SageMaker session
session = sagemaker.Session(boto3.session.Session(region_name=region))
role = sagemaker.get_execution_role()

print(f"SageMaker Role: {role}")

# Initialize the detector with default settings
detector = CorrosionDetector()
print(f"Model initialized on device: {detector.device}")

# Create a directory for sample images
os.makedirs('sample_images', exist_ok=True)

# For demo purposes, you should download or provide sample images
# Here we're assuming you have some sample images in the sample_images directory
# If you don't have sample images, you can download some from the internet

# List sample images
sample_images = os.listdir('sample_images')
print(f"Found {len(sample_images)} sample images: {sample_images}")

# Test the model on sample images
for image_file in sample_images:
    image_path = os.path.join('sample_images', image_file)
    
    # Open and display the image
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Original Image: {image_file}")
    plt.axis('off')
    
    # Detect corrosion
    detection_result = detector.detect(image)
    
    # Create a visualization
    title = f"Corrosion Detected (Confidence: {detection_result['confidence']:.2f})" if detection_result['has_corrosion'] else f"No Corrosion (Confidence: {1-detection_result['confidence']:.2f})"
    color = 'red' if detection_result['has_corrosion'] else 'green'
    
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.title(title, color=color)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Image: {image_file}")
    print(f"Detection Result: {'Corrosion Detected' if detection_result['has_corrosion'] else 'No Corrosion'}")
    print(f"Confidence: {detection_result['confidence']:.4f}")
    print("-" * 50)

# First, save the model to a local file
model_dir = '../models'
model_path = os.path.join(model_dir, 'corrosion_model.pth')

# Save the model weights
import torch
torch.save(detector.model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Upload the model to S3
s3_model_path = session.upload_data(model_path, bucket=config.S3_BUCKET, key_prefix=config.MODEL_S3_PATH)
print(f"Model uploaded to S3: {s3_model_path}")

# Create a PyTorch model for SageMaker
pytorch_model = PyTorchModel(
    model_data=s3_model_path,
    role=role,
    framework_version='1.12.0',
    py_version='py38',
    entry_point='inference.py',
    source_dir='../models',
)

# Deploy the model to an endpoint
predictor = pytorch_model.deploy(
    initial_instance_count=1,
    instance_type=config.INSTANCE_TYPE,
)

endpoint_name = predictor.endpoint_name
print(f"Model deployed to endpoint: {endpoint_name}")

def invoke_endpoint(image_path, endpoint_name):
    """
    Invoke the SageMaker endpoint with an image
    
    Args:
        image_path: Path to the image file
        endpoint_name: Name of the SageMaker endpoint
        
    Returns:
        The prediction result
    """
    # Read the image
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    # Create a SageMaker runtime client
    runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    # Invoke the endpoint
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='image/jpeg',
        Body=image_bytes
    )
    
    # Parse the response
    result = json.loads(response['Body'].read().decode())
    return result

# Test the endpoint with sample images
for image_file in sample_images:
    image_path = os.path.join('sample_images', image_file)
    
    # Open and display the image
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"Sample Image: {image_file}")
    plt.axis('off')
    plt.show()
    
    # Invoke the endpoint
    result = invoke_endpoint(image_path, endpoint_name)
    
    print(f"Image: {image_file}")
    print(f"Detection Result: {'Corrosion Detected' if result['has_corrosion'] else 'No Corrosion'}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("-" * 50)

# Process all sample images and create a summary
batch_results = []

for image_file in sample_images:
    image_path = os.path.join('sample_images', image_file)
    
    # Invoke the endpoint
    result = invoke_endpoint(image_path, endpoint_name)
    
    # Add to batch results
    batch_results.append({
        'filename': image_file,
        'has_corrosion': result['has_corrosion'],
        'confidence': result['confidence']
    })

# Display summary
corrosion_count = sum(1 for r in batch_results if r['has_corrosion'])
no_corrosion_count = len(batch_results) - corrosion_count

print("===== Batch Processing Summary =====")
print(f"Total images processed: {len(batch_results)}")
print(f"Images with corrosion: {corrosion_count}")
print(f"Images without corrosion: {no_corrosion_count}")
print("====================================")

# Plot the results
plt.figure(figsize=(10, 6))

# Prepare data for plotting
filenames = [r['filename'] for r in batch_results]
confidences = [r['confidence'] for r in batch_results]
colors = ['red' if r['has_corrosion'] else 'green' for r in batch_results]

# Create bar chart
plt.bar(filenames, confidences, color=colors)
plt.axhline(y=config.CONFIDENCE_THRESHOLD, color='black', linestyle='--', label=f'Threshold ({config.CONFIDENCE_THRESHOLD})')

plt.xlabel('Images')
plt.ylabel('Corrosion Confidence')
plt.title('Corrosion Detection Results')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
plt.tight_layout()
plt.legend()
plt.show()

# Delete the endpoint
predictor.delete_endpoint()
print(f"Endpoint {endpoint_name} deleted")

print("""
Conclusion:

In this demo, we've demonstrated how to:

1. Initialize the corrosion detection model
2. Test the model locally
3. Deploy the model to a SageMaker endpoint
4. Invoke the endpoint with sample images
5. Process a batch of images
6. Visualize the results

This workflow can be integrated into a production pipeline for automated corrosion detection 
in industrial settings using drone-captured images.
""") 