"""
Main application file for the Corrosion Detection System.
Provides a web interface for uploading images and viewing detection results.
Uses Claude 3 Sonnet multimodal AI for corrosion detection via AWS Bedrock.
"""

import os
import sys
import uuid
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from typing import Dict, Any, List

# Import project modules
import config
from src.ai_detector import detect_corrosion, visualize_detection_result

# Initialize Flask app
app = Flask(__name__, 
            static_folder='app/static',
            template_folder='app/templates')
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

def process_image(file) -> Dict[str, Any]:
    """
    Process an uploaded image file and detect corrosion.
    
    Args:
        file: The uploaded file object
        
    Returns:
        Dictionary with processing results
    """
    # Generate a unique filename
    ext = file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"{str(uuid.uuid4())}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    
    # Save the file
    file.save(filepath)
    
    # Detect corrosion using the configured method (AWS Bedrock by default)
    detection_result = detect_corrosion(filepath)
    
    # Create visualization
    result_image_path = visualize_detection_result(filepath, detection_result)

    print(f"Detected hazards: {detection_result.get('detected_hazards', [])}")
    
    # Prepare result information - Include ALL fields from detection_result
    result = {
        'filename': file.filename,
        'has_corrosion': detection_result['has_corrosion'],
        'has_hazards': detection_result.get('has_hazards', False),  # Add this
        'detected_hazards': detection_result.get('detected_hazards', []),  # Add this
        'confidence': detection_result['confidence'],
        'detailed_analysis': detection_result.get('detailed_analysis', ''),
        'original_image': filepath,
        'result_image': result_image_path
    }
    
    print(f"Result for {file.filename}: has_hazards={result['has_hazards']}, detected_hazards={len(result['detected_hazards'])}")
    
    return result

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload and hazard detection"""
    print("Upload endpoint called")  # Debug
    
    if 'files[]' not in request.files:
        print("No files[] in request.files")  # Debug
        flash('No file part')
        return redirect(request.url)
    
    files = request.files.getlist('files[]')
    print(f"Number of files received: {len(files)}")  # Debug
    
    if not files or files[0].filename == '':
        print("No files selected")  # Debug
        flash('No selected file')
        return redirect(request.url)
    
    results = []
    for file in files:
        print(f"Processing file: {file.filename}")  # Debug
        if file and allowed_file(file.filename):
            try:
                print(f"File allowed: {file.filename}")  # Debug
                result = process_image(file)
                results.append(result)
            except Exception as e:
                print(f"Error processing {file.filename}: {str(e)}")  # Debug
                flash(f"Error processing {file.filename}: {str(e)}")
        else:
            print(f"File not allowed: {file.filename}")  # Debug
            flash(f"File {file.filename} not allowed")
    
    print(f"Total results: {len(results)}")  # Debug
    
    return render_template('results.html', results=results)

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API endpoint for corrosion detection"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            result = process_image(file)
            return jsonify({
                'filename': result['filename'],
                'has_corrosion': result['has_corrosion'],
                'confidence': result['confidence'],
                'detailed_analysis': result.get('detailed_analysis', '')
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

# Add this route to your Flask app
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve files from the uploads folder (including subfolders)"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.context_processor
def inject_xsrf_token():
    """Make XSRF token available to templates"""
    if '_xsrf' in request.cookies:
        return dict(xsrf_token=request.cookies.get('_xsrf'))
    return dict(xsrf_token='')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 