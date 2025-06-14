#!/usr/bin/env python3
"""
fix-rag.py

This script adds additional metadata to JSON files in the rag-chunks directory.
It extracts "source" and "year" from the file level metadata and incorporates
them, along with the URL, into each individual chunk in the chunks array.
"""

import os
import json
import glob
import re


def process_file(file_path):
    """Process a single JSON file to add metadata to each chunk."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract metadata from the top level
        source = "wwdc"
        
        # Extract the year from the year field (e.g., "wwdc2023" -> "2023")
        year_match = re.search(r'wwdc(\d{4})', data.get('year', ''))
        year = year_match.group(1) if year_match else ""
        
        url = data.get('url', '')
        
        # Add the metadata to each chunk
        modified = False
        for chunk in data.get('chunks', []):
            # Only add if not already present
            if 'source' not in chunk:
                chunk['source'] = source
                modified = True
            if 'year' not in chunk:
                chunk['year'] = year
                modified = True
            if 'url' not in chunk:
                chunk['url'] = url
                modified = True
        
        # Write back if modified
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        return False
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    # Find all JSON files in the rag-chunks directory recursively
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rag_chunks_dir = os.path.join(base_dir, 'rag-chunks')
    
    if not os.path.exists(rag_chunks_dir):
        print(f"Error: rag-chunks directory not found at {rag_chunks_dir}")
        return
    
    # Find all JSON files recursively
    pattern = os.path.join(rag_chunks_dir, '**', '*.json')
    json_files = glob.glob(pattern, recursive=True)
    
    if not json_files:
        print(f"No JSON files found in {rag_chunks_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process.")
    
    # Process each file
    success_count = 0
    for file_path in json_files:
        if process_file(file_path):
            success_count += 1
            print(f"Processed: {os.path.relpath(file_path, base_dir)}")
    
    print(f"\nComplete! Successfully processed {success_count} "
          f"out of {len(json_files)} files.")


if __name__ == "__main__":
    main()
