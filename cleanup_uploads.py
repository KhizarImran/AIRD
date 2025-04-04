#!/usr/bin/env python3
"""
cleanup_uploads.py - Script to clean the uploads directory and its subdirectories
while preserving the directory structure.
"""

import os
import shutil
import sys

def clean_directory(directory_path):
    """
    Clean a directory by removing all files and subdirectories except
    for the 'results' directory if present.
    
    Args:
        directory_path: Path to the directory to clean
    """
    if not os.path.exists(directory_path):
        print(f"Directory does not exist: {directory_path}")
        return
        
    print(f"Cleaning directory: {directory_path}")
    
    # Get all items in the directory
    items = os.listdir(directory_path)
    
    for item in items:
        item_path = os.path.join(directory_path, item)
        
        # If this is the results directory, clean it recursively but preserve it
        if os.path.isdir(item_path) and item == 'results':
            print(f"Found results directory, cleaning its contents: {item_path}")
            clean_results_directory(item_path)
        else:
            # Otherwise, remove the file or directory
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                    print(f"Removed file: {item_path}")
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"Removed directory: {item_path}")
            except Exception as e:
                print(f"Error removing {item_path}: {e}")

def clean_results_directory(results_path):
    """
    Remove all files in the results directory.
    
    Args:
        results_path: Path to the results directory
    """
    for item in os.listdir(results_path):
        item_path = os.path.join(results_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
                print(f"Removed file: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Removed directory: {item_path}")
        except Exception as e:
            print(f"Error removing {item_path}: {e}")

def main():
    """Main function to run the script."""
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the uploads directory path relative to the script
    uploads_dir = os.path.join(script_dir, 'uploads')
    
    # Confirm before proceeding
    print(f"This will delete all files in {uploads_dir}, keeping only empty directories.")
    print("Are you sure you want to proceed? (y/n)")
    
    response = input().strip().lower()
    if response != 'y':
        print("Operation cancelled.")
        sys.exit(0)
    
    # Clean the uploads directory
    clean_directory(uploads_dir)
    print("Cleanup completed successfully.")

if __name__ == "__main__":
    main()