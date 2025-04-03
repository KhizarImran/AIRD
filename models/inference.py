"""
Inference script for SageMaker deployment of the corrosion detection model.
"""

import os
import json
import torch
import torchvision
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import io

# Default confidence threshold
DEFAULT_THRESHOLD = 0.7

def model_fn(model_dir):
    """
    Load the PyTorch model from the `model_dir` directory.
    
    Args:
        model_dir (str): The directory where model files are stored
        
    Returns:
        A loaded PyTorch model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model architecture
    model = resnet50(pretrained=False)
    
    # Modify the final fully connected layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 2),  # Two classes: corrosion or no corrosion
        torch.nn.Softmax(dim=1)
    )
    
    # Load the stored model weights
    model_path = os.path.join(model_dir, 'corrosion_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model = model.to(device)
    model.eval()
    
    return model

def input_fn(request_body, request_content_type):
    """
    Deserialize the input data for prediction.
    
    Args:
        request_body (str or bytes): The body of the request
        request_content_type (str): The content type of the request
        
    Returns:
        PIL Image: A PIL Image that can be used for prediction
    """
    if request_content_type == 'application/json':
        # Handle JSON input - expect a base64 encoded image
        request = json.loads(request_body)
        image_data = request.get('image')
        import base64
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        return image
    
    elif request_content_type == 'image/jpeg' or request_content_type == 'image/png':
        # Handle binary image input
        image = Image.open(io.BytesIO(request_body))
        return image
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Generate prediction from the input data.
    
    Args:
        input_data: PIL Image to perform inference on
        model (PyTorch model): The loaded PyTorch model
        
    Returns:
        dict: Prediction result
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Convert image to appropriate format for model
    image_tensor = preprocess(input_data).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = output.cpu().numpy()[0]
    
    # Get the confidence score for corrosion (class 1)
    corrosion_confidence = float(probabilities[1])
    
    # Determine if corrosion is detected based on confidence threshold
    has_corrosion = corrosion_confidence >= DEFAULT_THRESHOLD
    
    return {
        'has_corrosion': bool(has_corrosion),
        'confidence': corrosion_confidence
    }

def output_fn(prediction, response_content_type):
    """
    Serialize the prediction result into the desired response content type.
    
    Args:
        prediction (dict): The prediction result
        response_content_type (str): The content type of the response
        
    Returns:
        str: The serialized prediction
    """
    if response_content_type == 'application/json':
        return json.dumps(prediction)
    else:
        # Default to JSON
        return json.dumps(prediction) 