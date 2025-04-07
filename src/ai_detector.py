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
Analyze this image for signs of corrosion and other infrastructure hazards. Focus specifically on identifying:

1. CORROSION: Rust patches, discoloration, pitting, flaking on metal surfaces
2. LOOSE_COMPONENTS: Unsecured fasteners, hanging/detached parts, misalignments
3. STRUCTURAL_INTEGRITY: Bent components, cracks, deformations, compromised elements
4. ELECTRICAL_HAZARDS: Exposed wiring, improper connections, damaged electrical components
5. ENVIRONMENTAL_THREATS: Debris accumulation, vegetation interference, water damage
6. SAFETY_EQUIPMENT_ISSUES: Missing/damaged guards, signs, safety devices

For EACH detected hazard, provide:
- Category: One of the six categories above (exactly as written)
- Coordinates: [x1, y1, x2, y2] as fractions of image dimensions (between 0.0-1.0)
- Severity: 0.1-1.0 score based on urgency/danger level
- Description: Brief description of the specific issue detected

Format your response as a JSON object:
{
  "has_corrosion": boolean,
  "confidence": float,
  "detailed_analysis": string,
  "detected_hazards": [
    {
      "category": string,
      "coordinates": [x1, y1, x2, y2],
      "severity": float,
      "description": string
    }
  ]
}

Response should contain ONLY this JSON object, no additional text.
"""
MULTI_HAZARD_PROMPT_TEMPLATE = """
Analyze this image for infrastructure hazards. Focus specifically on identifying:

1. CORROSION: Rust patches, discoloration, pitting, flaking on metal surfaces
2. LOOSE_COMPONENTS: Unsecured fasteners, hanging/detached parts, misalignments
3. STRUCTURAL_INTEGRITY: Bent components, cracks, deformations, compromised elements
4. ELECTRICAL_HAZARDS: Exposed wiring, improper connections, damaged electrical components
5. ENVIRONMENTAL_THREATS: Debris accumulation, vegetation interference, water damage
6. SAFETY_EQUIPMENT_ISSUES: Missing/damaged guards, signs, safety devices

For EACH detected hazard, provide:
- Category: One of the six categories above (exactly as written)
- Coordinates: [x1, y1, x2, y2] as fractions of image dimensions (between 0.0-1.0)
- Severity: 0.1-1.0 score based on urgency/danger level
- Description: Brief description of the specific issue detected

Format your response as a JSON object:
{
  "has_hazards": boolean,
  "confidence": float,
  "detected_hazards": [
    {
      "category": string,
      "coordinates": [x1, y1, x2, y2],
      "severity": float,
      "description": string
    }
  ]
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
    Main function to detect corrosion and other hazards.
    Keeps the original function name for compatibility.
    
    Args:
        image_path: Path to the image file
        use_aws_bedrock: Whether to use AWS Bedrock
        
    Returns:
        Dictionary with detection results including multiple hazard types
    """
    # If not explicitly specified, use the config setting
    if use_aws_bedrock is None:
        use_aws_bedrock = config.USE_AWS_BEDROCK
    
    # Get detection results using the original methods
    if use_aws_bedrock:
        # Use the existing detect_corrosion_aws_bedrock function
        result = detect_corrosion_aws_bedrock(image_path)
    else:
        # Use the existing detect_corrosion_claude function
        result = detect_corrosion_claude(image_path)
    
    # Apply post-processing to enhance the results
    result = refine_hazard_boxes(result)
    
    return result

def refine_hazard_boxes(detection_result: Dict[str, Any], min_size: float = 0.02) -> Dict[str, Any]:
    """
    Cleans up and improves hazard detection boxes.
    
    Args:
        detection_result: Raw detection results
        min_size: Minimum box size as fraction of image
        
    Returns:
        Improved detection results
    """
    # Initialize detected_hazards if not present
    if 'detected_hazards' not in detection_result:
        detection_result['detected_hazards'] = []
    
    # For backward compatibility, create detected_hazards from corrosion result
    if detection_result.get('has_corrosion', False) and not detection_result['detected_hazards']:
        # If it's a corrosion result without explicit hazards, create a hazard entry
        detection_result['detected_hazards'].append({
            'category': 'CORROSION',
            'coordinates': [0.1, 0.1, 0.9, 0.9],  # Default box if no specific coordinates
            'severity': detection_result.get('confidence', 0.5),
            'description': detection_result.get('detailed_analysis', 'Corrosion detected')
        })
        
    # Set has_hazards flag based on existing data
    detection_result['has_hazards'] = detection_result.get('has_corrosion', False) or len(detection_result.get('detected_hazards', [])) > 0
    
    return detection_result

def visualize_detection_result(image_path: str, detection_result: Dict[str, Any]) -> str:
    """
    Enhanced visualization function that handles multiple hazard types.
    Keeps the original function name for compatibility.
    
    Args:
        image_path: Path to the original image
        detection_result: Dictionary with detection results
        
    Returns:
        Web-accessible path to the output visualization image
    """
    from PIL import Image, ImageDraw, ImageFont
    
    # Open the original image
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Define colors for different hazard categories
    category_colors = {
        "CORROSION": (255, 0, 0),               # Red
        "LOOSE_COMPONENTS": (255, 165, 0),      # Orange
        "STRUCTURAL_INTEGRITY": (0, 0, 255),    # Blue
        "ELECTRICAL_HAZARDS": (255, 255, 0),    # Yellow
        "ENVIRONMENTAL_THREATS": (0, 128, 0),   # Green
        "SAFETY_EQUIPMENT_ISSUES": (128, 0, 128)  # Purple
    }
    
    # Default color for undefined categories
    default_color = (128, 128, 128)  # Gray
    
    # Get detection results
    has_hazards = detection_result.get('has_hazards', False) or detection_result.get('has_corrosion', False)
    
    # Create a border color based on detection
    border_color = (255, 0, 0, 180) if has_hazards else (0, 255, 0, 180)
    
    # Draw border
    border_width = 10
    for i in range(border_width):
        draw.rectangle(
            [(i, i), (width - i - 1, height - i - 1)],
            outline=border_color[:3]
        )
    
    # Try to load fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 20)
        label_font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
    
    # Draw hazard regions
    if 'detected_hazards' in detection_result:
        for i, hazard in enumerate(detection_result['detected_hazards']):
            # Get category and color
            category = hazard.get('category', '').upper()
            color = category_colors.get(category, default_color)
            
            if 'coordinates' in hazard and len(hazard['coordinates']) == 4:
                try:
                    # Get coordinates and convert from relative (0-1) to absolute pixels
                    x1, y1, x2, y2 = hazard['coordinates']
                    x1, y1 = int(x1 * width), int(y1 * height)
                    x2, y2 = int(x2 * width), int(y2 * height)
                    
                    # Skip invalid boxes
                    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                        continue
                        
                    # Draw rectangle with category-specific color
                    box_thickness = 3
                    for j in range(box_thickness):
                        draw.rectangle([(x1+j, y1+j), (x2-j, y2-j)], outline=color)
                    
                    # Create label with hazard number and category
                    severity = hazard.get('severity', 0)
                    label = f"#{i+1}: {category.replace('_', ' ')}"
                    if severity > 0:
                        label += f" ({severity:.2f})"
                        
                    # Draw label background
                    text_bbox = draw.textbbox((x1, y1-25), label, font=label_font)
                    draw.rectangle(
                        [
                            (text_bbox[0] - 5, text_bbox[1] - 5),
                            (text_bbox[2] + 5, text_bbox[3] + 5)
                        ],
                        fill=(0, 0, 0, 180)
                    )
                    
                    # Draw label text
                    draw.text((x1, y1-25), label, fill=color, font=label_font)
                    
                except (ValueError, TypeError, IndexError) as e:
                    print(f"Error processing hazard {i}: {e}")
                    continue
                    
    # Add overall status text
    if has_hazards:
        # Count hazards by category
        category_counts = {}
        if 'detected_hazards' in detection_result:
            for hazard in detection_result['detected_hazards']:
                category = hazard.get('category', '').upper()
                if category:
                    category_counts[category] = category_counts.get(category, 0) + 1
        
        # Create status text based on what was found
        status_lines = []
        for category, count in category_counts.items():
            category_name = category.replace('_', ' ')
            status_lines.append(f"{category_name}: {count}")
        
        status_text = "HAZARDS DETECTED"
    else:
        status_text = "NO HAZARDS DETECTED"
    
    # Add a background for text
    padding = 10
    text_x, text_y = 20, 20
    text_bbox = draw.textbbox((text_x, text_y), status_text, font=title_font)
    draw.rectangle(
        [
            (text_bbox[0] - padding, text_bbox[1] - padding),
            (text_bbox[2] + padding, text_bbox[3] + padding)
        ],
        fill=(0, 0, 0, 180)
    )
    
    # Draw status text
    text_color = border_color[:3]
    draw.text((text_x, text_y), status_text, fill=text_color, font=title_font)
    
    # Add category counts if hazards were detected
    if has_hazards and status_lines:
        y_offset = text_bbox[3] + 15
        for line in status_lines:
            category = line.split(':')[0].strip()
            color = category_colors.get(category.replace(' ', '_'), default_color)
            
            line_bbox = draw.textbbox((text_x, y_offset), line, font=label_font)
            draw.rectangle(
                [
                    (line_bbox[0] - padding, line_bbox[1] - padding),
                    (line_bbox[2] + padding, line_bbox[3] + padding)
                ],
                fill=(0, 0, 0, 180)
            )
            draw.text((text_x, y_offset), line, fill=color, font=label_font)
            y_offset = line_bbox[3] + 10
    
    # Add a legend
    if has_hazards:
        legend_x = width - 200
        legend_y = 20
        
        # Draw a background
        legend_height = 30 + (len(category_colors) * 25)
        draw.rectangle(
            [(legend_x - 10, legend_y - 10), (legend_x + 190, legend_y + legend_height)],
            fill=(0, 0, 0, 180)
        )
        
        # Draw legend title
        draw.text((legend_x, legend_y), "HAZARD TYPES:", fill=(255, 255, 255), font=label_font)
        y_offset = legend_y + 30
        
        # Draw entries for each category
        for category, color in category_colors.items():
            # Draw color swatch
            draw.rectangle([(legend_x, y_offset), (legend_x + 15, y_offset + 15)], fill=color)
            
            # Draw category name
            category_name = category.replace('_', ' ')
            draw.text((legend_x + 25, y_offset), category_name, fill=(255, 255, 255), font=label_font)
            y_offset += 25
    
    # Save the output image
    output_dir = os.path.join(os.path.dirname(image_path), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.basename(image_path)
    output_filename = f"result_{base_filename}"
    output_path = os.path.join(output_dir, output_filename)
    
    image.save(output_path)
    
    # Return web path
    web_path = f"uploads/results/{output_filename}"
    return web_path