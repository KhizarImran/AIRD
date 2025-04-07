"""
Functional module for detecting corrosion in images using multimodal AI models (Claude 3).
This module handles the interaction with the AI models via AWS Bedrock.
"""

import os
import json
import base64
import requests
from typing import Dict, Any, Union, Tuple
from io import BytesIO
from PIL import Image
import boto3
from dotenv import load_dotenv
import config

# Load environment variables
load_dotenv()

# Constants and configuration
CONFIDENCE_THRESHOLD = config.CONFIDENCE_THRESHOLD
CLAUDE_PROMPT_TEMPLATE = """
Analyze this image for signs of corrosion on industrial structures or telecommunications towers.

IMPORTANT: Be extremely precise about the exact location of visible corrosion. Focus only on areas showing:
- Rust patches with orange/brown discoloration
- Pitting, flaking, or scaling of metal surfaces
- Areas where paint has deteriorated exposing corroded metal
- Structural components with visible degradation

For each corrosion area, provide:
1. Precise bounding box coordinates [x1, y1, x2, y2] as fractions of image dimensions (values between 0.0-1.0)
2. Severity score (0.1-1.0) based on the apparent corrosion extent
3. Brief description of the specific corrosion type observed

Only identify regions where corrosion is clearly visible - do not mark areas that are merely aged or discolored but not corroded.

Format your response as a JSON object:
{
  "has_corrosion": boolean,
  "confidence": float,
  "corrosion_regions": [
    {"coordinates": [x1, y1, x2, y2], "severity": float, "description": "specific description of this corrosion area"}
  ],
  "detailed_analysis": "overall assessment of the structure's corrosion status"
}

Response should contain ONLY this JSON object, no additional text.
"""

def get_api_key() -> str:
    """Get the API key from environment variables."""
    # Try to get API key from environment variables
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    return api_key

def encode_image_to_base64(image_path: str) -> str:
    """
    Encodes an image to base64 for API transmission.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_bedrock_client():
    """
    Create and configure an AWS Bedrock client.
    
    Returns:
        Configured boto3 bedrock-runtime client
    """
    # Check if explicit credentials are provided
    if config.AWS_ACCESS_KEY_ID and config.AWS_SECRET_ACCESS_KEY:
        session = boto3.Session(
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION
        )
        return session.client(service_name="bedrock-runtime")
    else:
        # Use default credentials from environment or IAM role
        return boto3.client(
            service_name="bedrock-runtime",
            region_name=config.AWS_REGION
        )

def detect_corrosion_claude(image_path: str) -> Dict[str, Any]:
    """
    Detect corrosion in an image using Claude 3 Sonnet directly via Anthropic API.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with detection results:
            - has_corrosion: Boolean indicating if corrosion is detected
            - confidence: Confidence score for the prediction
            - detailed_analysis: Detailed text analysis from Claude
    """
    api_key = get_api_key()
    client = anthropic.Anthropic(api_key=api_key)
    
    # Read and encode the image
    base64_image = encode_image_to_base64(image_path)
    
    # Create the message with the image
    try:
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            temperature=0,
            system="You are an expert in materials science and corrosion detection. Analyze images for signs of corrosion.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": CLAUDE_PROMPT_TEMPLATE
                        }
                    ]
                }
            ]
        )
        
        # Parse the JSON response
        response_text = message.content[0].text
        response_json = json.loads(response_text)
        
        # Ensure all required fields are present
        if not all(k in response_json for k in ["has_corrosion", "confidence", "detailed_analysis"]):
            raise ValueError("Response missing required fields")
            
        return response_json
        
    except Exception as e:
        print(f"Error in Claude API call: {e}")
        # Return a default response
        return {
            "has_corrosion": False,
            "confidence": 0.0,
            "detailed_analysis": f"Error in analysis: {str(e)}"
        }

def detect_corrosion_aws_bedrock(image_path: str) -> Dict[str, Any]:
    """
    Detect corrosion in an image using Claude 3 via AWS Bedrock.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with detection results
    """
    # Get AWS Bedrock client
    bedrock_runtime = get_bedrock_client()
    
    # Read and encode the image
    base64_image = encode_image_to_base64(image_path)
    
    # Create the request payload
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0,
        "system": "You are an expert in materials science and corrosion detection. Analyze images for signs of corrosion.",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": CLAUDE_PROMPT_TEMPLATE
                    }
                ]
            }
        ]
    }
    
    try:
        # Call AWS Bedrock with Claude model
        response = bedrock_runtime.invoke_model(
            body=json.dumps(payload),
            modelId=config.MODEL_NAME,
            contentType="application/json",
            accept="application/json"
        )
        
        # Parse the response
        response_body = json.loads(response.get("body").read())
        response_text = response_body.get("content")[0].get("text")
        response_json = json.loads(response_text)
        
        return response_json
        
    except Exception as e:
        print(f"Error in AWS Bedrock API call: {e}")
        return {
            "has_corrosion": False,
            "confidence": 0.0,
            "detailed_analysis": f"Error in analysis: {str(e)}"
        }

def detect_corrosion(image_path: str, use_aws_bedrock: bool = None) -> Dict[str, Any]:
    """
    Main function to detect corrosion in an image using the preferred method.
    
    Args:
        image_path: Path to the image file
        use_aws_bedrock: Whether to use AWS Bedrock. If None, uses the config setting.
        
    Returns:
        Dictionary with detection results
    """
    # If not explicitly specified, use the config setting
    if use_aws_bedrock is None:
        use_aws_bedrock = config.USE_AWS_BEDROCK
        
    if use_aws_bedrock:
        return detect_corrosion_aws_bedrock(image_path)
    else:
        return detect_corrosion_claude(image_path)

def visualize_detection_result(image_path: str, detection_result: Dict[str, Any]) -> str:
    """
    Create a visualization of the corrosion detection results with bounding boxes.
    
    Args:
        image_path: Path to the original image
        detection_result: Dictionary with detection results
        
    Returns:
        Web-accessible path to the output visualization image
    """
    # Import needed modules at the top of your file, not here
    from PIL import Image, ImageDraw, ImageFont
    
    # Open the original image
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # Get detection results
    has_corrosion = detection_result['has_corrosion']
    confidence = detection_result['confidence']
    
    # Create a border color based on detection
    border_color = (255, 0, 0, 180) if has_corrosion else (0, 255, 0, 180)
    
    # Draw border
    width, height = image.size
    border_width = 10
    for i in range(border_width):
        draw.rectangle(
            [(i, i), (width - i - 1, height - i - 1)],
            outline=border_color[:3]  # Remove alpha for PIL
        )
    
    # Draw corrosion regions if they exist
    if has_corrosion and 'corrosion_regions' in detection_result:
        for i, region in enumerate(detection_result['corrosion_regions']):
            if 'coordinates' in region:
                # Get coordinates and convert from relative (0-1) to absolute pixels
                x1, y1, x2, y2 = region['coordinates']
                x1, y1 = int(x1 * width), int(y1 * height)
                x2, y2 = int(x2 * width), int(y2 * height)
                
                # Draw rectangle around the corrosion area
                box_color = (255, 0, 0)  # Red for corrosion
                box_thickness = 3
                for j in range(box_thickness):
                    draw.rectangle([(x1+j, y1+j), (x2-j, y2-j)], outline=box_color)
                
                # Add region number and severity if available
                label = f"#{i+1}"
                if 'severity' in region:
                    label += f" ({region['severity']:.2f})"
                    
                # Draw label background
                font_size = 16
                try:
                    font = ImageFont.truetype("arial.ttf", font_size)
                except IOError:
                    font = ImageFont.load_default()
                    
                text_bbox = draw.textbbox((x1, y1-20), label, font=font)
                draw.rectangle(
                    [
                        (text_bbox[0] - 5, text_bbox[1] - 5),
                        (text_bbox[2] + 5, text_bbox[3] + 5)
                    ],
                    fill=(0, 0, 0, 180)
                )
                
                # Draw label text
                draw.text((x1, y1-20), label, fill=(255, 255, 255), font=font)
    
    # Add text with results
    status_text = "CORROSION DETECTED" if has_corrosion else "NO CORROSION"
    confidence_text = f"Confidence: {confidence:.2f}"
    
    # Get appropriate font
    font_size = 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate text position
    text_x = 20
    text_y = 20
    
    # Add a background for text
    padding = 10
    text_bbox = draw.textbbox((text_x, text_y), status_text, font=font)
    draw.rectangle(
        [
            (text_bbox[0] - padding, text_bbox[1] - padding),
            (text_bbox[2] + padding, text_bbox[3] + padding)
        ],
        fill=(0, 0, 0, 180)
    )
    
    # Draw text
    draw.text((text_x, text_y), status_text, fill=border_color[:3], font=font)
    
    # Add confidence text
    conf_text_y = text_bbox[3] + 10
    conf_bbox = draw.textbbox((text_x, conf_text_y), confidence_text, font=font)
    draw.rectangle(
        [
            (conf_bbox[0] - padding, conf_bbox[1] - padding),
            (conf_bbox[2] + padding, conf_bbox[3] + padding)
        ],
        fill=(0, 0, 0, 180)
    )
    draw.text((text_x, conf_text_y), confidence_text, fill=(255, 255, 255), font=font)
    
    # Save the output image
    output_dir = os.path.join(os.path.dirname(image_path), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.basename(image_path)
    output_filename = f"result_{base_filename}"
    output_path = os.path.join(output_dir, output_filename)
    
    image.save(output_path)
    
    # Instead of returning the full file path, return a web URL path
    # This is the key change: construct a URL path that will work with your Flask routes
    web_path = f"uploads/results/{output_filename}"
    
    return web_path