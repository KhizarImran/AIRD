"""
Corrosion detector module that handles the detection of corrosion in images.
Uses a pre-trained model via SageMaker for inference.
"""

import os
import json
import numpy as np
import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import boto3
import sagemaker
from PIL import Image

import config


class CorrosionDetector:
    """Class for detecting corrosion in images using a pre-trained model."""
    
    def __init__(self, model_path=None):
        """
        Initialize the corrosion detector.
        
        Args:
            model_path: Path to the model weights. If None, will use a pre-trained model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Initialize model
        self.model = self._initialize_model(model_path)
        
        # Configure AWS
        self.region = config.AWS_REGION
        self.s3_bucket = config.S3_BUCKET
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        
    def _initialize_model(self, model_path=None):
        """
        Initialize the model for corrosion detection.
        
        Args:
            model_path: Path to model weights or None
            
        Returns:
            The initialized model
        """
        # Start with a pre-trained ResNet50
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify the final fully connected layer for binary classification (corrosion or no corrosion)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 2),  # Two classes: corrosion or no corrosion
            torch.nn.Softmax(dim=1)
        )
        
        # Load model weights if specified
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        model = model.to(self.device)
        model.eval()  # Set to evaluation mode
        
        return model
    
    def detect(self, image):
        """
        Detect corrosion in an image.
        
        Args:
            image: PIL Image or path to image
            
        Returns:
            Dictionary with detection results:
                - has_corrosion: Boolean indicating if corrosion is detected
                - confidence: Confidence score for the prediction
        """
        # Convert string path to PIL Image if needed
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = outputs.cpu().numpy()[0]
            
        # Get the confidence score for corrosion (class 1)
        corrosion_confidence = float(probs[1])
        
        # Determine if corrosion is detected based on confidence threshold
        has_corrosion = corrosion_confidence >= self.confidence_threshold
        
        return {
            'has_corrosion': has_corrosion,
            'confidence': corrosion_confidence
        }
    
    def deploy_to_sagemaker(self):
        """
        Deploy the model to SageMaker for production use.
        
        Returns:
            The SageMaker endpoint name
        """
        # Create SageMaker session
        session = sagemaker.Session(boto3.session.Session(region_name=self.region))
        
        # Save model to a local file first
        model_path = 'models/corrosion_model.pth'
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        
        # Upload model to S3
        s3_model_path = session.upload_data(model_path, bucket=self.s3_bucket, key_prefix=config.MODEL_S3_PATH)
        
        # Create SageMaker model
        role = config.SAGEMAKER_ROLE
        pytorch_model = sagemaker.PyTorchModel(
            model_data=s3_model_path,
            role=role,
            framework_version='1.12.0',
            py_version='py38',
            entry_point='inference.py',
            source_dir='models',
        )
        
        # Deploy model to endpoint
        predictor = pytorch_model.deploy(
            initial_instance_count=1,
            instance_type=config.INSTANCE_TYPE,
        )
        
        return predictor.endpoint_name 