import os
import pathlib
import json
from typing import List
import requests
import re
import logging
import click

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# === CONFIG ===
INPUT_FOLDER = "./transcripts"
OUTPUT_FOLDER = "./rag-chunks"

# Filepath for the checkpoint file
CHECKPOINT_FILE = "checkpoint.json"


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


def lmstudio_chunker_via_rest(
    markdown_text: str, model_name: str = "qwen/qwen3-14b"
) -> List[str]:
    """
    Use the LM Studio REST API to perform chunking of the markdown text.
    """
    logger.info(f"Using model: {model_name}")
    url = "http://localhost:11234/v1/chat/completions"
    prompt = (
        """
/no_think
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
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32768,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "stop": None,
        "gpu": 1.0,
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


def safe_title(filename: str) -> str:
    """Convert filename to a safe folder name (no extension, spaces -> underscores)."""
    logger.debug(f"Converting filename '{filename}' to a safe title.")
    return pathlib.Path(filename).stem.replace(" ", "_")


def save_progress(processed_files):
    """Save the list of processed files to a checkpoint file."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed_files": processed_files}, f)


def load_progress():
    """Load the list of processed files from the checkpoint file."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f).get("processed_files", [])
    return []


def load_progress_filecheck(input_folder: str, output_folder: str) -> set:
    """Load processed files by comparing source and destination files."""
    processed_files = set()

    input_files = set(os.listdir(input_folder))
    output_files = {
        os.path.splitext(f)[0] for f in os.listdir(output_folder)
    }  # Remove extensions

    for file in input_files:
        file_stem = os.path.splitext(file)[0]  # Get the stem of the file
        if file_stem in output_files:
            processed_files.add(file)

    return processed_files


def process_markdown_file(
    file_path: str, output_folder: str, model_name: str, processed_files: list
):
    """Helper function to process a single markdown file."""
    filename = os.path.basename(file_path)
    logger.info(f"Processing file: {filename}")
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

    # Check if the file size exceeds the context window limit
    max_context_window_size = 32768  # Example context window size in characters
    if os.path.getsize(file_path) > max_context_window_size:
        logger.error(
            f"File '{filename}' exceeds the context window size and will be skipped."
        )
        return

    talk_title = safe_title(filename)
    chunks = lmstudio_chunker_via_rest(content, model_name)
    logger.info(f"Generated {len(chunks)} chunks for file: {filename}")

    talk_output_dir = os.path.join(output_folder, model_name)
    os.makedirs(talk_output_dir, exist_ok=True)
    logger.info(f"Created output directory: {talk_output_dir}")

    output_dict = {
        "year": year,
        "title": title,
        "url": url,
        "model": model_name,  # Add dynamic model name to metadata
        "chunks": chunks,
    }
    chunks_json_path = os.path.join(talk_output_dir, f"{talk_title}.json")
    with open(chunks_json_path, "w", encoding="utf-8") as json_file:
        json.dump(output_dict, json_file, indent=2)
    logger.info(f"Saved chunks to: {chunks_json_path}")

    # Mark the file as processed and save progress
    processed_files.append(filename)
    save_progress(processed_files)


def process_all_markdown_files(
    input_folder: str, output_folder: str, model_name: str = "qwen/qwen3-14b"
):
    logger.info(
        f"Processing all markdown files in folder: {input_folder} with model: {model_name}"
    )

    # Load previously processed files
    # processed_files = load_progress()
    processed_files = load_progress_filecheck(
        input_folder=input_folder, output_folder=output_folder
    )

    for filename in os.listdir(input_folder):
        if not filename.endswith(".md"):
            logger.debug(f"Skipping non-markdown file: {filename}")
            continue

        if filename in processed_files:
            logger.info(f"Skipping already processed file: {filename}")
            continue

        file_path = os.path.join(input_folder, filename)
        process_markdown_file(file_path, output_folder, model_name, processed_files)


def process_single_markdown_file(
    input_folder: str,
    output_folder: str,
    target_file: str,
    model_name: str = "qwen/qwen3-14b",
):
    """Process a single markdown file."""
    # Handle absolute or relative file paths
    if not os.path.isabs(target_file) and not target_file.startswith(input_folder):
        file_path = os.path.join(input_folder, target_file)
    else:
        file_path = target_file

    filename = os.path.basename(file_path)  # Extract the filename

    if not os.path.exists(file_path):
        logger.error(f"File '{file_path}' does not exist.")
        return

    # Check if the corresponding output file exists
    output_file_path = os.path.join(output_folder, model_name, f"{os.path.splitext(filename)[0]}.json")
    if os.path.exists(output_file_path):
        logger.info(f"Skipping already processed file: {filename}")
        return

    logger.info(f"Processing single markdown file: {file_path} with model: {model_name}")
    process_file(file_path, model_name)


@click.command()
@click.option(
    "--model_name",
    default="qwen/qwen3-14b",
    help="Name of the model to use for processing.",
)
@click.option(
    "--restart",
    is_flag=True,
    default=False,
    help="Restart processing from the beginning, ignoring checkpoints.",
)
@click.option(
    "--file",
    "target_file",
    default=None,
    help="Process a single markdown file instead of all files.",
)
def main(model_name: str, restart: bool, target_file: str):
    """Main function to process markdown files with command-line arguments."""
    if restart:
        logger.info("Restarting processing and clearing checkpoint file.")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)

    if target_file:
        logger.info(f"Processing single file: {target_file} with model: {model_name}")
        process_single_markdown_file(
            INPUT_FOLDER, OUTPUT_FOLDER, target_file, model_name
        )
    else:
        logger.info(f"Starting processing with model: {model_name}")
        process_all_markdown_files(INPUT_FOLDER, OUTPUT_FOLDER, model_name)


if __name__ == "__main__":
    main()
