import json
import logging
import os
import re
import time

import click
from mlx_lm import generate, load

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_metadata(lines):
    """
    Pull out YEAR, TITLE, and URL from the document header.
    """
    meta = {}
    for line in lines:
        if line.startswith("YEAR:"):
            meta["year"] = line.split(":", 1)[1].strip().lower()
        elif line.startswith("TITLE:"):
            meta["title"] = line.split(":", 1)[1].strip()
        elif line.startswith("URL:"):
            meta["url"] = line.split(":", 1)[1].strip()
    return meta


def clean_content(lines):
    """
    Join and strip extra whitespace, normalize newlines.
    """
    text = "".join(lines).strip()
    # collapse multiple blank lines
    text = re.sub(r"\n{3,}", r"\n\n", text)
    return text


def slugify(text):
    """
    Simple slugify for filenames.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def chunk_and_summarize(body_text, model, tokenizer, max_tokens, overlap, thinking):
    """
    Use MLXLM to split into chunks and summarize each.
    """
    # split out code fence blocks as discrete chunks
    # regex to capture ```...``` blocks including their fences
    code_split_pattern = re.compile(r"(```[\s\S]+?```)")
    raw_segments = code_split_pattern.split(body_text)
    chunks = []
    for seg in raw_segments:
        if seg.startswith("```"):
            # treat entire code block as its own chunk
            chunks.append(seg)
        else:
            # non-code text, will chunk below
            text_to_chunk = seg
            start = 0
            seg_len = len(text_to_chunk)
            while start < seg_len:
                end = min(start + max_tokens * 4, seg_len)
                chunk = text_to_chunk[start:end]
                chunks.append(chunk)
                start += max_tokens * 4 - overlap * 4
    # now `chunks` holds both code and text chunks

    results = []
    for chunk in chunks:
        # build a prompt that asks for JSON output
        instruction = (
            'Please output a JSON object with keys "title" (a concise 3-6 word headline) '
            'and "summary" (a 1-2 sentence summary) for the following text chunk.'
        )
        conversation = [
            {"role": "system", "content": "You extract titles and summaries in JSON."},
            {"role": "user", "content": f"{instruction}\n\nText chunk:\n{chunk}"},
        ]
        if not thinking:
            conversation.append({"role": "user", "content": "/no_think"})
        prompt = tokenizer.apply_chat_template(
            conversation=conversation, add_generation_prompt=True
        )

        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=512,
            verbose=False,
        )
        # remove any think tags and whitespace
        clean = response.replace("<think>", "").replace("</think>", "").strip()
        try:
            parsed = json.loads(clean)
            title = parsed.get("title", "").strip()
            summary = parsed.get("summary", "").strip()
        except json.JSONDecodeError:
            # fallback: split on first newline
            parts = clean.split("\n", 1)
            title = parts[0].strip() if parts else ""
            summary = parts[1].strip() if len(parts) > 1 else ""
        results.append(
            {"title": title.strip(), "summary": summary.strip(), "content": chunk}
        )
    return results


@click.command()
@click.option(
    "--path",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Directory path containing the input file",
)
@click.option("--filename", "-f", required=True, help="Name of the file to process")
@click.option(
    "--model", "-m", default="mlx-community/Qwen3-8B-4bit", help="MLX model identifier"
)
@click.option(
    "--max-tokens", default=32768, help="Maximum tokens per chunk (up to 40000)"
)
@click.option("--overlap", default=50, help="Token overlap between chunks")
@click.option(
    "--thinking",
    is_flag=True,
    default=False,
    help="Enable thinking mode (default: False)",
)
def prep_rag(path, filename, model, max_tokens, overlap, thinking):
    """
    Read a document, extract metadata, chunk, summarize, and output JSON for RAG.
    """
    start_time = time.time()
    logger.info(f"Starting RAG prep for file: {filename} in path: {path}")

    # Load the model and tokenizer
    model_obj, tokenizer = load(path_or_hf_repo=model)

    full_path = os.path.join(path, filename)
    with open(full_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # extract metadata from top of file
    meta = extract_metadata(lines[:20])
    if not meta.get("year") or not meta.get("title") or not meta.get("url"):
        click.echo(
            "Error: Could not find YEAR, TITLE, or URL in the first 20 lines.", err=True
        )
        raise click.Abort()

    # clean up entire content
    body_text = clean_content(lines)

    # chunk and summarize
    logger.info(f"Chunking and summarizing with {model}...")
    chunks = chunk_and_summarize(
        body_text, model_obj, tokenizer, max_tokens, overlap, thinking
    )

    # assemble result
    result = {
        "year": meta["year"],
        "title": meta["title"],
        "url": meta["url"],
        "model": model,
        "chunks": chunks,
    }

    # create output filename
    base = f"{meta['year']}_{slugify(meta['title'])}.json"
    out_path = os.path.join(path, base)
    with open(out_path, "w", encoding="utf-8") as out:
        json.dump(result, out, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    logger.info(f"Completed RAG prep in {mins:02d}:{secs:02d}")

    logger.info(f"Output written to {out_path}")


if __name__ == "__main__":
    prep_rag()
