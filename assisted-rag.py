import json
import logging
import os
import pathlib
import re
from typing import List, Protocol

import click
import requests

tokenizer = None


class LLMInterface(Protocol):
    def chat(self, messages: List[dict], options: dict = None) -> str: ...


class LMStudio:
    def __init__(self, model_name: str = "qwen3:8b"):
        self.model_name = model_name
        self.api_url = "http://localhost:11234/v1/chat/completions"

    def chat(self, messages: List[dict], options: dict = None) -> str:
        if options is None:
            options = {
                "num_ctx": 40960,
                "num_predict": 2048,
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 20,
            }
        data = {
            "model": self.model_name,
            "messages": messages,
            "options": options,
        }
        try:
            response = requests.post(self.api_url, json=data)
            response.raise_for_status()
            result_json = response.json()
            return result_json["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LMStudio.chat failed: {e}")
            raise


# === MLXLM class for LLMInterface compatibility ===
class MLXLM:
    def __init__(self, model_path: str = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"):
        from mlx_lm import load

        self.model, self.tokenizer = load(model_path)
        self.prompt_cache = None

    def chat(self, messages: List[dict], options: dict = None) -> str:
        from mlx_lm import generate
        from mlx_lm.models.cache import make_prompt_cache

        if self.prompt_cache is None:
            self.prompt_cache = make_prompt_cache(self.model)

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            verbose=False,
            prompt_cache=self.prompt_cache,
        )
        return response


def get_tokenizer(model_name):
    global tokenizer
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8b")
    return tokenizer


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


def extract_topics(text: str, llm: LLMInterface) -> List[dict]:
    """
    Extracts a list of topics and their descriptions from the input text using
    the provided LLMInterface.
    """
    logger.info("Extracting topics from transcript.")
    prompt = f"""
<context>
/no_think
You are a technical summarizer skilled at identifying the key themes in 
developer-focused transcripts.
</context>
<task>
List the main topics covered in the following transcript. For each topic, 
provide:
- A short title (2–5 words)
- A one-sentence description summarizing what is discussed

Return only a JSON array in the following format:
[
  {{
    "topic": "Title of Topic",
    "description": "One sentence summary of the topic"
  }},
  ...
]
</task>
<input>{text}</input>
"""
    try:
        response = llm.chat([{"role": "user", "content": prompt}])
        topics = extract_json_from_text(response)
        return topics
    except Exception as e:
        logger.error(f"Error extracting topics: {e}")
        return []


def extract_json_from_text(text: str):
    """
    Extract the first JSON array found in the given text using regex.
    Strips markdown fences and whitespace before parsing.
    """
    logger.debug("Extracting JSON from text.")

    # Strip <think>*</think> tags if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip ```json ... ``` fencing if present
    if text.startswith("```json"):
        text = text[len("```json") :].strip()
    if text.endswith("```"):
        text = text[: -len("```")].strip()

    # Now extract the JSON array (starting with [ and ending with ])
    json_array_pattern = re.compile(r"\[\s*{.*?}\s*\]", re.DOTALL)
    match = json_array_pattern.search(text)
    if not match:
        raise json.JSONDecodeError("No JSON array found", text, 0)

    json_text = match.group(0)
    return json.loads(json_text)


# def extract_json_from_text(text: str):
#     """
#     Extract the first JSON array found in the given text using regex.
#     Returns the parsed JSON object or raises json.JSONDecodeError if none found.
#     """
#     logger.debug("Extracting JSON from text.")
#     # Regex to find JSON array (starting with [ and ending with ])
#     json_array_pattern = re.compile(r"\[\s*{.*}\s*\]", re.DOTALL)
#     match = json_array_pattern.search(text)
#     if not match:
#         raise json.JSONDecodeError("No JSON array found", text, 0)
#     json_text = match.group(0)
#     return json.loads(json_text)


def lmstudio_chunker_via_rest(
    markdown_text: str, model_name: str = "qwen3:8b"
) -> List[str]:
    """
    Use the LM Studio REST API to perform chunking of the markdown text.
    """
    logger.info(f"Using model: {model_name}")
    prompt = """
<context>
        You are an expert technical editor.
        /no_think
</context>
<task>
Your task is to take the provided text (a transcript, article, or technical 
document) and break it into coherent, self-contained chunks. These chunk are 
used for RAG.

If you can not provide a chunk. Do not synthesize new information.

Each chunk should:
- Focus on a single topic or subtopic
- Be no more than ~300–500 words
- Include a short title and a brief 1–2 sentence summary
- Contain enough context to make sense on its own. Do not summarize the content.
  it should be the raw transcript.

Output format:
[
  {
    "title": "<Chunk Title>",
    "summary": "<Brief summary>",
    "content": "<Chunked body text>"
  },
  ...
]
</task>
<input>{markdown_text}</input>"""

    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "num_ctx": 40960,
            "num_predict": 32768,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
        },
    }
    try:
        llm = LMStudio(model_name)
        completion_text = llm.chat(
            [{"role": "user", "content": prompt}], options=data["options"]
        )
        chunks = extract_json_from_text(completion_text)
    except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
        logger.error(f"Error during LM Studio REST API call or JSON parsing: {e}")
        logger.warning(f"Raw LLM output (last 2000 chars):\n{completion_text[-2000:]}")
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


def split_text_by_tokens(text, max_tokens=2048, overlap=200, model_name="qwen3:8b"):
    token_ids = get_tokenizer(model_name).encode(text)
    chunks = []

    for start in range(0, len(token_ids), max_tokens - overlap):
        end = min(start + max_tokens, len(token_ids))
        chunk_text = get_tokenizer(model_name).decode(token_ids[start:end])
        chunks.append(chunk_text)
    logger.info(f"Split text into {len(chunks)} chunks based on token count.")
    if not chunks:
        logger.warning("No chunks created from the text.")
        return []
    return chunks


def split_by_tokens(
    content: str, max_tokens: int = 20000, model_name: str = "qwen3:8b"
) -> List[str]:
    """
    Split content into chunks based on token count.
    """
    return split_text_by_tokens(
        content, max_tokens=max_tokens, overlap=200, model_name=model_name
    )


def chunker(markdown_text: str, llm: LLMInterface) -> List[str]:
    """
    Chunk the markdown text using the provided LLMInterface.
    """
    logger.info(f"Using chunker with LLM: {llm.__class__.__name__}")
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert technical editor. Your task is to take the provided text "
                "(a transcript, article, or technical document) and break it into coherent, "
                "self-contained chunks. These chunks are used for Retrieval-Augmented Generation (RAG). "
                "Each chunk should focus on a single topic or subtopic, be no more than ~300–500 words, "
                "include a short title and a brief 1–2 sentence summary, and contain enough context to "
                "make sense on its own. Do not summarize the content—include the raw content itself. "
                "If you cannot provide a chunk, do not fabricate or synthesize content."
            ),
        },
        {"role": "user", "content": markdown_text},
    ]
    try:
        completion_text = llm.chat(messages)
        chunks = extract_json_from_text(completion_text)
    except Exception as e:
        logger.error(f"Error during chunker call or JSON parsing: {e}")
        logger.warning(
            f"Raw LLM output (last 2000 chars):\n{completion_text[-2000:] if 'completion_text' in locals() else ''}"
        )
        return []
    return chunks


def process_with_retry(
    content: str, llm: LLMInterface, model_name: str, max_retries: int = 3
) -> List[dict]:
    """
    Process content with progressive splitting on failure.
    Automatically splits content into smaller chunks on API errors.
    """
    logger.info(
        f"Starting process_with_retry. Initial content length: {len(content)} characters"
    )
    current_chunks = [content]
    results = []

    for retry in range(max_retries):
        success = True
        new_chunks = []

        for chunk in current_chunks:
            try:
                # Try to process the chunk
                chunk_result = chunker(chunk, llm)
                if chunk_result:  # Only add if we got valid results
                    results.extend(chunk_result)
                    logger.info(f"Chunk result count: {len(chunk_result)}")
                else:
                    # If no results, split the chunk further
                    logger.warning(
                        f"No results returned for chunk. Splitting further. "
                        f"Retry {retry + 1}/{max_retries}"
                    )
                    # Split into smaller chunks based on token estimate
                    new_chunk_size = 20000 - (retry * 5000)  # Reduce size
                    new_chunks.extend(
                        split_by_tokens(
                            chunk, max_tokens=new_chunk_size, model_name=model_name
                        )
                    )
                    success = False
            except Exception as e:
                # If processing fails, split the chunk further
                logger.warning(
                    f"Processing failed, splitting chunk further: {e}. "
                    f"Retry {retry + 1}/{max_retries}"
                )
                # Split into smaller chunks based on token estimate
                new_chunk_size = 20000 - (retry * 5000)  # Reduce size
                new_chunks.extend(
                    split_by_tokens(
                        chunk, max_tokens=new_chunk_size, model_name=model_name
                    )
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


def split_large_file(content: str, max_size: int, model_name: str) -> List[str]:
    """
    Split a large file into smaller chunks that fit within the context
    window size. Uses token-based splitting for better accuracy.
    """
    # Use token-based splitting instead of character-based
    return split_by_tokens(content, max_tokens=20000, model_name=model_name)


def process_markdown_file(
    file_path: str, output_folder: str, model_name: str, processed_files: list
):
    """Helper function to process a single markdown file."""
    import time

    start_time = time.time()
    filename = os.path.basename(file_path)
    logger.info(f"Processing file: {filename}")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    def extract_code_blocks_from_markdown(content: str) -> List[dict]:
        pattern = re.compile(
            r"--- Code Sample \d+ ---\s+\*\*Time\*\*: (.*?)\s+\*\*Title\*\*: (.*?)\n\n```swift\n(.*?)```",
            re.DOTALL,
        )
        matches = pattern.findall(content)
        return [
            {"timestamp": ts.strip(), "title": title.strip(), "code": code.strip()}
            for ts, title, code in matches
        ]

    logger.info(f"Loaded content length: {len(content)} characters")

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

    # Initialize LLM
    llm = LMStudio(model_name=model_name)

    # Process content with progressive splitting
    logger.info(f"Processing content for file: {filename}")
    topics = extract_topics(content, llm=llm)
    if topics:
        logger.info("Extracted Topics:")
        for i, topic in enumerate(topics, 1):
            logger.info(f"{i}. {topic['topic']}: {topic['description']}")
    else:
        logger.warning("No topics extracted.")

    # # Use our new token-based approach with retries
    # all_chunks = process_with_retry(content, llm, model_name, max_retries=3)

    # # If processing completely failed, log an error
    # if not all_chunks:
    #     logger.error(f"Failed to process file '{filename}' after multiple attempts")
    #     all_chunks = []

    # logger.info(f"Generated {len(all_chunks)} chunks for file: {filename}")

    # # Extract source as "wwdc" from the year field
    # source = "wwdc"
    # # Extract numeric year from the year field (e.g., "wwdc2023" -> "2023")
    # numeric_year = ""
    # year_match = re.search(r"wwdc(\d{4})", year)
    # if year_match:
    #     numeric_year = year_match.group(1)

    # # Add parsed code blocks from markdown to all_chunks
    # code_blocks = extract_code_blocks_from_markdown(content)
    # for block in code_blocks:
    #     all_chunks.append(
    #         {
    #             "title": f"Code: {block['title']}",
    #             "summary": f"Code sample from the session – {block['title']}",
    #             "content": block["code"],
    #             "type": "code",
    #             "source": source,
    #             "year": numeric_year,
    #             "url": url,
    #         }
    #     )
    # logger.info(f"Appended {len(code_blocks)} code chunks from markdown.")

    # # Add metadata to each chunk
    # for chunk in all_chunks:
    #     chunk["source"] = source
    #     chunk["year"] = numeric_year
    #     chunk["url"] = url

    # talk_output_dir = os.path.join(output_folder, model_name)
    # os.makedirs(talk_output_dir, exist_ok=True)
    # logger.info(f"Created output directory: {talk_output_dir}")

    # output_dict = {
    #     "year": year,
    #     "title": title,
    #     "url": url,
    #     "model": model_name,  # Add dynamic model name to metadata
    #     "chunks": all_chunks,
    # }
    # chunks_json_path = os.path.join(talk_output_dir, f"{safe_title(filename)}.json")
    # with open(chunks_json_path, "w", encoding="utf-8") as json_file:
    #     json.dump(output_dict, json_file, indent=2)
    # elapsed_time = time.time() - start_time
    # minutes = int(elapsed_time // 60)
    # seconds = int(elapsed_time % 60)
    # logger.info(f"Processing time for {filename}: {minutes}:{seconds:02d}")
    # logger.info(f"Saved chunks to: {chunks_json_path}")

    # # Mark the file as processed and save progress
    # processed_files.append(filename)
    # save_progress(processed_files)


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
    force: bool = False,
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
    output_file_path = os.path.join(
        output_folder, model_name, f"{os.path.splitext(filename)[0]}.json"
    )
    if os.path.exists(output_file_path) and not force:
        logger.info(f"Skipping already processed file: {filename}")
        return
    elif os.path.exists(output_file_path) and force:
        logger.info(f"Force flag enabled, reprocessing file: {filename}")

    logger.info(
        f"Processing single markdown file: {file_path} "
        f"with model: {model_name}" + (f" (force={force})" if force else "")
    )

    processed_files = list(load_progress_filecheck(INPUT_FOLDER, OUTPUT_FOLDER))

    logger.info(f"Found {len(processed_files)} already processed files.")

    process_file(file_path, model_name, force=force)


def process_file(file_path: str, model_name: str, force: bool = False):
    """Wrapper for processing a single markdown file."""
    process_markdown_file(
        file_path=file_path,
        output_folder=OUTPUT_FOLDER,
        model_name=model_name,
        processed_files=list(load_progress_filecheck(INPUT_FOLDER, OUTPUT_FOLDER)),
    )


@click.command()
@click.option(
    "--model_name",
    default="qwen3:14b",
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
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force reprocessing of the file even if output already exists.",
)
@click.option(
    "--topics-only",
    is_flag=True,
    default=False,
    help="Parse the input file and print the topics to the console",
)
def main(
    model_name: str,
    restart: bool,
    target_file: str,
    force: bool,
    topics_only: bool,
):
    """Main function to process markdown files with command-line arguments."""
    if restart:
        logger.info("Restarting processing and clearing checkpoint file.")
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)

    if target_file:
        logger.info(
            f"Processing single file: {target_file} "
            f"with model: {model_name} (force={force})"
        )

        # Get list of already processed files to check against
        processed_files = list(load_progress_filecheck(INPUT_FOLDER, OUTPUT_FOLDER))
        if processed_files:
            logger.info(f"Found {len(processed_files)} already processed files.")
        process_single_markdown_file(
            INPUT_FOLDER,
            OUTPUT_FOLDER,
            target_file,
            model_name,
            force=force,
        )
    else:
        logger.info(f"Starting processing with model: {model_name}")
        process_all_markdown_files(INPUT_FOLDER, OUTPUT_FOLDER, model_name)


if __name__ == "__main__":
    main()
