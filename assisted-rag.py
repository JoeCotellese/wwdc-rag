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
        "max_context_length": 32768,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0,
        "stop": None,
        "gpu": 2.0,
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


def normalize_folder_path(folder: str) -> str:
    """Normalize folder paths to remove redundant structures."""
    return os.path.normpath(folder)


def load_progress_filecheck(input_folder: str, output_folder: str) -> list:
    """Load processed files by comparing source and destination files."""
    processed_files = []

    # Normalize folder paths
    input_folder = normalize_folder_path(input_folder)
    output_folder = normalize_folder_path(output_folder)

    input_files = set(os.listdir(input_folder))
    output_files = set()

    # Traverse subdirectories in the output folder
    for root, _, files in os.walk(output_folder):
        for file in files:
            output_files.add(os.path.splitext(file)[0])  # Add stem of the file

    for file in input_files:
        file_stem = os.path.splitext(file)[0]  # Get the stem of the file
        if file_stem in output_files:
            processed_files.append(file)

    return processed_files


def split_by_tokens(content: str, max_tokens: int = 20000) -> List[str]:
    """
    Split content based on an estimated token count rather than character
    count.
    Uses a conservative estimate of characters per token.
    """
    # Estimate: 1 token ≈ 4 characters for English text (rough approximation)
    char_per_token = 4
    max_chars = max_tokens * char_per_token

    chunks = []
    while len(content) > max_chars:
        # Try to split at paragraph boundary
        split_point = content.rfind("\n\n", 0, max_chars)
        if split_point == -1:
            # Try to split at line boundary
            split_point = content.rfind("\n", 0, max_chars)
        if split_point == -1:
            # Try to split at sentence boundary
            split_point = content.rfind(". ", 0, max_chars)
        if split_point == -1:
            # Last resort: split at character
            split_point = max_chars

        chunks.append(content[:split_point].strip())
        content = content[split_point:].strip()

    if content:  # Add remaining content if not empty
        chunks.append(content)

    return chunks


def process_with_retry(
    content: str,
    model_name: str,
    max_retries: int = 3
) -> List[dict]:
    """
    Process content with progressive splitting on failure.
    Automatically splits content into smaller chunks on API errors.
    """
    current_chunks = [content]
    results = []

    for retry in range(max_retries):
        success = True
        new_chunks = []

        for chunk in current_chunks:
            try:
                # Try to process the chunk
                chunk_result = lmstudio_chunker_via_rest(chunk, model_name)
                if chunk_result:  # Only add if we got valid results
                    results.extend(chunk_result)
                else:
                    # If no results, split the chunk further
                    logger.warning(
                        f"No results returned for chunk. Splitting further. "
                        f"Retry {retry+1}/{max_retries}"
                    )
                    # Split into smaller chunks based on token estimate
                    new_chunk_size = 20000 - (retry * 5000)  # Reduce size
                    new_chunks.extend(
                        split_by_tokens(chunk, max_tokens=new_chunk_size)
                    )
                    success = False
            except Exception as e:
                # If processing fails, split the chunk further
                logger.warning(
                    f"Processing failed, splitting chunk further: {e}. "
                    f"Retry {retry+1}/{max_retries}"
                )
                # Split into smaller chunks based on token estimate
                new_chunk_size = 20000 - (retry * 5000)  # Reduce size
                new_chunks.extend(
                    split_by_tokens(chunk, max_tokens=new_chunk_size)
                )
                success = False

        if success or not new_chunks:
            # If all chunks processed successfully or no more
            # splitting possible
            break

        # If any chunk failed, replace current_chunks with new_chunks
        # for the next retry
        current_chunks = new_chunks

    if not results:
        logger.error("Failed to process content after maximum retries")

    return results


def split_large_file(content: str, max_size: int) -> List[str]:
    """
    Split a large file into smaller chunks that fit within the context
    window size. Uses token-based splitting for better accuracy.
    """
    # Use token-based splitting instead of character-based
    return split_by_tokens(content, max_tokens=20000)


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
        logger.info(
            f"Extracted metadata - Year: {year}, Title: {title}, URL: {url}"
        )
    else:
        logger.warning(f"Metadata not found in file: {filename}")

    # Process content with progressive splitting
    logger.info(f"Processing content for file: {filename}")
    
    # Use our new token-based approach with retries
    all_chunks = process_with_retry(content, model_name, max_retries=3)
    
    # If processing completely failed, log an error
    if not all_chunks:
        logger.error(
            f"Failed to process file '{filename}' after multiple attempts"
        )
        all_chunks = []
        
    logger.info(f"Generated {len(all_chunks)} chunks for file: {filename}")

    talk_output_dir = os.path.join(output_folder, model_name)
    os.makedirs(talk_output_dir, exist_ok=True)
    logger.info(f"Created output directory: {talk_output_dir}")

    output_dict = {
        "year": year,
        "title": title,
        "url": url,
        "model": model_name,  # Add dynamic model name to metadata
        "chunks": all_chunks,
    }
    chunks_json_path = os.path.join(
        talk_output_dir, f"{safe_title(filename)}.json"
    )
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
        f"Processing all markdown files in folder: {input_folder} "
        f"with model: {model_name}"
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
        process_markdown_file(
            file_path=file_path,
            output_folder=output_folder,
            model_name=model_name,
            processed_files=processed_files,
        )


def process_single_markdown_file(
    input_folder: str,
    output_folder: str,
    target_file: str,
    model_name: str = "qwen/qwen3-14b",
):
    """Process a single markdown file."""
    # Handle absolute or relative file paths
    if not os.path.isabs(target_file) and not target_file.startswith(
        input_folder
    ):
        file_path = os.path.join(input_folder, target_file)
    else:
        file_path = target_file

    filename = os.path.basename(file_path)  # Extract the filename

    if not os.path.exists(file_path):
        logger.error(f"File '{file_path}' does not exist.")
        return

    # Check if the corresponding output file exists
    output_file_path = os.path.join(
        output_folder, model_name, f"{os.path.splitext(filename)[0]}.json"
    )
    if os.path.exists(output_file_path):
        logger.info(f"Skipping already processed file: {filename}")
        return

    logger.info(
        f"Processing single markdown file: {file_path} "
        f"with model: {model_name}"
    )

    processed_files = list(
        load_progress_filecheck(INPUT_FOLDER, OUTPUT_FOLDER)
    )

    # Use processed_files in the logic
    if processed_files:
        logger.info("Loaded processed files successfully.")
        for file in processed_files:
            logger.debug(f"Already processed: {file}")

    process_file(file_path, model_name)


def process_file(file_path: str, model_name: str):
    """Wrapper for processing a single markdown file."""
    process_markdown_file(
        file_path=file_path,
        output_folder=OUTPUT_FOLDER,
        model_name=model_name,
        processed_files=list(
            load_progress_filecheck(INPUT_FOLDER, OUTPUT_FOLDER)
        ),
    )


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
        logger.info(
            f"Processing single file: {target_file} with model: {model_name}"
        )

        # Get list of already processed files to check against
        processed_files = list(
            load_progress_filecheck(INPUT_FOLDER, OUTPUT_FOLDER)
        )
        if processed_files:
            logger.info(
                f"Found {len(processed_files)} already processed files."
            )
    else:
        logger.info(f"Starting processing with model: {model_name}")
        process_all_markdown_files(INPUT_FOLDER, OUTPUT_FOLDER, model_name)


if __name__ == "__main__":
    main()
