import os
import pathlib
import json
from typing import List
import requests
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === CONFIG ===
INPUT_FOLDER = "./transcripts"
OUTPUT_FOLDER = "./rag-chunks"

def extract_json_from_text(text: str):
    """
    Extract the first JSON array found in the given text using regex.
    Returns the parsed JSON object or raises json.JSONDecodeError if none found.
    """
    logger.debug("Extracting JSON from text.")
    # Regex to find JSON array (starting with [ and ending with ])
    json_array_pattern = re.compile(r"\[\s*{.*?}\s*\]", re.DOTALL)
    match = json_array_pattern.search(text)
    if not match:
        raise json.JSONDecodeError("No JSON array found", text, 0)
    json_text = match.group(0)
    return json.loads(json_text)

def lmstudio_chunker_via_rest(markdown_text: str, model_name: str = "qwen/qwen3-14b") -> List[str]:
    """
    Use the LM Studio REST API to perform chunking of the markdown text.
    """
    logger.info(f"Using model: {model_name}")
    url = "http://localhost:11234/v1/chat/completions"
    prompt = (
        """
You are an expert technical editor.

Your task is to take the provided text (a transcript, article, or technical document) and break it into coherent, self-contained chunks.

Each chunk should:
- Focus on a single topic or subtopic
- Be no more than ~300–500 words
- Include a short title and a brief 1–2 sentence summary
- Contain enough context to make sense on its own

Output format:
[
  {
    "title": "<Chunk Title>",
    "summary": "<Brief summary>",
    "content": "<Chunked body text>"
  },
  ...
]
"""
        + markdown_text
    )
    data = {
        "model": model_name,  # Use dynamic model name
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 32768,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "stop": None,
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        logger.info("Received response from LM Studio API.")
        result_json = response.json()
        # The API response format may vary; adjust accordingly
        # Now expecting the content in result_json['choices'][0]['message']['content']
        completion_text = result_json["choices"][0]["message"]["content"]
        chunks = extract_json_from_text(completion_text)
    except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
        logger.error(f"Error during LM Studio REST API call or JSON parsing: {e}")
        return []

    return chunks


def generate_metadata(chunk_text: str) -> dict:
    """
    Stub function to generate metadata for a chunk.
    """
    logger.debug("Generating metadata for a chunk.")
    return {
        "chunk_title": "Observation in SwiftUI",
        "keywords": ["SwiftUI", "Observation", "WWDC", "Apple"],
        "summary": "This chunk discusses the Observation feature in SwiftUI introduced at WWDC 2023.",
    }


def safe_title(filename: str) -> str:
    """Convert filename to a safe folder name (no extension, spaces -> underscores)."""
    logger.debug(f"Converting filename '{filename}' to a safe title.")
    return pathlib.Path(filename).stem.replace(" ", "_")


def process_all_markdown_files(input_folder: str, output_folder: str, model_name: str = "qwen/qwen3-14b"):
    logger.info(f"Processing all markdown files in folder: {input_folder} with model: {model_name}")
    for filename in os.listdir(input_folder):
        if not filename.endswith(".md"):
            logger.debug(f"Skipping non-markdown file: {filename}")
            continue

        logger.info(f"Processing file: {filename}")
        file_path = os.path.join(input_folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract YEAR, TITLE, URL from header using regex
        year, title, url = "", "", ""
        header_match = re.search(
            r"YEAR:\s*(.+)\nTITLE:\s*(.+)\nURL:\s*(.+)\n", content, re.IGNORECASE
        )
        if header_match:
            year = header_match.group(1).strip()
            title = header_match.group(2).strip()
            url = header_match.group(3).strip()
            logger.info(f"Extracted metadata - Year: {year}, Title: {title}, URL: {url}")
        else:
            logger.warning(f"Metadata not found in file: {filename}")

        talk_title = safe_title(filename)
        chunks = lmstudio_chunker_via_rest(content, model_name)
        logger.info(f"Generated {len(chunks)} chunks for file: {filename}")

        talk_output_dir = os.path.join(output_folder, talk_title)
        os.makedirs(talk_output_dir, exist_ok=True)
        logger.info(f"Created output directory: {talk_output_dir}")

        output_dict = {
            "year": year,
            "title": title,
            "url": url,
            "model": model_name,  # Add dynamic model name to metadata
            "chunks": chunks
        }
        chunks_json_path = os.path.join(talk_output_dir, "chunks.json")
        with open(chunks_json_path, "w", encoding="utf-8") as json_file:
            json.dump(output_dict, json_file, indent=2)
        logger.info(f"Saved chunks to: {chunks_json_path}")

def process_single_markdown_file(input_folder: str, output_folder: str, target_file: str, model_name: str = "qwen/qwen3-14b"):
    logger.info(f"Processing single markdown file: {target_file} with model: {model_name}")
    file_path = os.path.join(input_folder, target_file)
    if not os.path.exists(file_path):
        logger.error(f"File '{file_path}' does not exist.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract YEAR, TITLE, URL from header using regex
    year, title, url = "", "", ""
    header_match = re.search(
        r"YEAR:\s*(.+)\nTITLE:\s*(.+)\nURL:\s*(.+)\n", content, re.IGNORECASE
    )
    if header_match:
        year = header_match.group(1).strip()
        title = header_match.group(2).strip()
        url = header_match.group(3).strip()
        logger.info(f"Extracted metadata - Year: {year}, Title: {title}, URL: {url}")
    else:
        logger.warning(f"Metadata not found in file: {target_file}")

    talk_title = safe_title(target_file)
    chunks = lmstudio_chunker_via_rest(content, model_name)
    logger.info(f"Generated {len(chunks)} chunks for file: {target_file}")

    talk_output_dir = os.path.join(output_folder, talk_title)
    os.makedirs(talk_output_dir, exist_ok=True)
    logger.info(f"Created output directory: {talk_output_dir}")

    output_dict = {
        "year": year,
        "title": title,
        "url": url,
        "model": model_name,  # Add dynamic model name to metadata
        "chunks": chunks
    }
    chunks_json_path = os.path.join(talk_output_dir, "chunks.json")
    with open(chunks_json_path, "w", encoding="utf-8") as json_file:
        json.dump(output_dict, json_file, indent=2)
    logger.info(f"Saved chunks to: {chunks_json_path}")


# if __name__ == "__main__":
#     # Temporarily process only the specified file
#     TARGET_FILE = "wwdc2025-245-What_s_new_in_Swift_-_WWDC25_-_Videos_-_Apple_Developer.md"
#     process_single_markdown_file(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_FILE)

if __name__ == "__main__":
    model_name = "qwen/qwen3-14b"  # Define the model name here
    logger.info(f"Starting processing of markdown files with model: {model_name}")
    process_all_markdown_files(INPUT_FOLDER, OUTPUT_FOLDER, model_name)
    logger.info("Finished processing markdown files.")
    process_all_markdown_files(INPUT_FOLDER, OUTPUT_FOLDER)
    logger.info("Finished processing markdown files.")
