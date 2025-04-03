"""
Utility functions for image processing and visualization.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def preprocess_image(image_path):
    """
    Preprocess an image for input to the corrosion detection model.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image object
    """
    # Load image with PIL
    image = Image.open(image_path).convert('RGB')
    return image

def visualize_results(image_path, detection_result):
    """
    Create a visualization of the corrosion detection results.
    
    Args:
        image_path: Path to the original image
        detection_result: Dictionary with detection results from CorrosionDetector
        
    Returns:
        Path to the output visualization image
    """
    # Load the original image
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    
    # Get detection results
    has_corrosion = detection_result['has_corrosion']
    confidence = detection_result['confidence']
    
    # Create a border to indicate corrosion detection status
    width, height = image.size
    border_width = 10
    
    # Red border for corrosion detection, green border for no corrosion
    color = (255, 0, 0) if has_corrosion else (0, 255, 0)
    
    # Draw border
    for i in range(border_width):
        draw.rectangle(
            [(i, i), (width - i - 1, height - i - 1)],
            outline=color
        )
    
    # Add text annotation
    status_text = "CORROSION DETECTED" if has_corrosion else "NO CORROSION"
    confidence_text = f"Confidence: {confidence:.2f}"
    
    # Get appropriate font (default if no specific font is available)
    font_size = 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate text position
    text_width, text_height = draw.textsize(status_text, font=font)
    text_x = (width - text_width) // 2
    text_y = 20
    
    # Add a background rectangle for text
    rect_padding = 10
    draw.rectangle(
        [(text_x - rect_padding, text_y - rect_padding),
         (text_x + text_width + rect_padding, text_y + text_height + rect_padding)],
        fill=(0, 0, 0, 180)
    )
    
    # Draw text
    draw.text((text_x, text_y), status_text, fill=color, font=font)
    draw.text((text_x, text_y + text_height + 10), confidence_text, fill=(255, 255, 255), font=font)
    
    # Save the output image
    output_dir = os.path.join(os.path.dirname(image_path), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.basename(image_path)
    output_filename = f"result_{base_filename}"
    output_path = os.path.join(output_dir, output_filename)
    
    image.save(output_path)
    return output_path

def create_heatmap(image_path, detection_result, cam_image=None):
    """
    Create a heatmap visualization showing areas of corrosion.
    
    Args:
        image_path: Path to the original image
        detection_result: Dictionary with detection results
        cam_image: Optional class activation map image (numpy array)
        
    Returns:
        Path to the output heatmap image
    """
    # Load the original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # If no CAM image is provided, create a simple placeholder
    if cam_image is None:
        # Create a simple heatmap based on confidence
        has_corrosion = detection_result['has_corrosion']
        confidence = detection_result['confidence']
        
        if has_corrosion:
            # Create a dummy heatmap (centered hotspot)
            h, w = image.shape[:2]
            cam_image = np.zeros((h, w))
            
            # Create a gradient from center
            y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
            mask = x*x + y*y <= min(h, w)**2 / 4
            center_y, center_x = h//2, w//2
            
            # Apply radial gradient
            for i in range(h):
                for j in range(w):
                    dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                    max_dist = np.sqrt(center_y**2 + center_x**2)
                    cam_image[i, j] = max(0, 1 - dist / max_dist) * confidence
        else:
            # No corrosion - blank heatmap
            cam_image = np.zeros(image.shape[:2])
    
    # Resize the heatmap to match the image size if needed
    if cam_image.shape[:2] != image.shape[:2]:
        cam_image = cv2.resize(cam_image, (image.shape[1], image.shape[0]))
    
    # Normalize the heatmap
    cam_image = (cam_image - cam_image.min()) / (cam_image.max() - cam_image.min() + 1e-8)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_image), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image
    alpha = 0.4
    superimposed_img = heatmap * alpha + image * (1 - alpha)
    superimposed_img = superimposed_img / np.max(superimposed_img)
    superimposed_img = np.uint8(255 * superimposed_img)
    
    # Save the output image
    output_dir = os.path.join(os.path.dirname(image_path), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.basename(image_path)
    output_filename = f"heatmap_{base_filename}"
    output_path = os.path.join(output_dir, output_filename)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(superimposed_img)
    plt.title('Corrosion Heatmap')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path 