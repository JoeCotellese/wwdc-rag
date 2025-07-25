import json
import logging
import os
import re
import time

import click
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_metadata(lines):
    meta = {}
    for line in lines:
        if line.startswith("YEAR:"):
            year_string = line.split(":", 1)[1].strip().lower()
            meta["year"] = re.sub(r"^wwdc", "", year_string)
            meta["wwdc"] = year_string
        elif line.startswith("TITLE:"):
            meta["title"] = line.split(":", 1)[1].strip()
        elif line.startswith("URL:"):
            meta["url"] = line.split(":", 1)[1].strip()
    return meta


def clean_content(lines):
    text = "".join(lines).strip()
    text = re.sub(r"\n{3,}", r"\n\n", text)
    return text


def slugify(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def chunk_by_paragraphs(content, tokenizer=None, max_tokens=256):
    if tokenizer is None:
        raise ValueError("A tokenizer must be provided.")

    paragraphs = content.strip().split("\n")
    chunks = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        token_count = len(tokenizer.encode(para, add_special_tokens=True))
        if token_count < 12:
            logger.info(f"skipping short chunk: {para}")
            continue
        logger.debug(f"processing {para[:10]} with {token_count} tokens")
        if token_count <= max_tokens:
            chunks.append(para)
        else:
            logger.info(f"Paragraph > {max_tokens}. Splitting by sentence")
            # fallback: sentence-based splitting inside the paragraph
            sentences = re.split(r"(?<=[.!?])\s+", para)
            buffer = ""
            for sentence in sentences:
                candidate = f"{buffer} {sentence}".strip()
                if (
                    len(tokenizer.encode(candidate, add_special_tokens=True))
                    > max_tokens
                ):
                    if buffer:
                        chunks.append(buffer.strip())
                    buffer = sentence
                else:
                    buffer = candidate
            if buffer:
                chunks.append(buffer.strip())
    return chunks


def deterministic_chunk(body_text, wwdc: str, year: str, title: str, url: str):
    code_pattern = re.compile(
        r"--- Code Sample \d+ ---[\s\S]+?```[\s\S]+?```", re.MULTILINE
    )
    code_chunks = code_pattern.findall(body_text)

    content_match = re.search(r"CONTENT:\s*(.*?)\s*CODE SAMPLES:", body_text, re.DOTALL)
    content = content_match.group(1).strip() if content_match else ""
    paragraph_chunks = chunk_by_paragraphs(content, tokenizer=model.tokenizer)

    chunks = []

    if body_text.strip():
        chunks.append(
            {
                "title": "Main Content",
                "summary": "Narrative content from the session.",
                "type": "transcript",
                "content": body_text,
            }
        )

    for i, paragraph in enumerate(paragraph_chunks, start=1):
        chunks.append(
            {
                "title": title,
                "wwdc": wwdc,
                "year": year,
                "url": url,
                "chunk_number": i,
                "type": "chunk",
                "content": paragraph,
            }
        )

    for code_block in code_chunks:
        title_match = re.search(r"\*\*Title\*\*: (.+)", code_block)
        title = title_match.group(1).strip() if title_match else "Code Sample"
        summary = f"Code example: {title}."
        chunks.append(
            {
                "title": title,
                "summary": summary,
                "type": "code",
                "content": code_block.strip(),
            }
        )

    return chunks


@click.command()
@click.option(
    "--file",
    "-f",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a single file to process",
)
@click.option(
    "--dir",
    "-d",
    type=click.Path(exists=True, file_okay=False),
    help="Path to a directory of files to process",
)
@click.option(
    "--model", "-m", default="deterministic", help="Model identifier (for bookkeeping)"
)
@click.option(
    "--max-tokens",
    default=32768,
    help="Maximum tokens per chunk (unused in this script)",
)
@click.option(
    "--overlap", default=50, help="Token overlap between chunks (unused in this script)"
)
@click.option(
    "--thinking",
    is_flag=True,
    default=False,
    help="Enable thinking mode (unused in this script)",
)
@click.option(
    "--outdir",
    default="./rag-chunks",
    type=click.Path(file_okay=False),
    help="Directory to write output JSON files",
)
def prep_rag(file, dir, model, max_tokens, overlap, thinking, outdir):
    start_time = time.time()

    if file:
        logger.info(f"Starting RAG prep for file: {file}")
        input_files = [file]
    elif dir:
        logger.info(f"Starting RAG prep for directory: {dir}")
        input_files = [
            os.path.join(dir, fname)
            for fname in os.listdir(dir)
            if fname.lower().endswith(".md")
        ]
    else:
        click.echo("Error: You must provide either --file or --dir.", err=True)
        raise click.Abort()

    os.makedirs(outdir, exist_ok=True)

    for full_path in input_files:
        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        meta = extract_metadata(lines[:20])
        if not meta.get("year") or not meta.get("title") or not meta.get("url"):
            click.echo(
                "Error: Could not find YEAR, TITLE, or URL in the first 20 lines.",
                err=True,
            )
            raise click.Abort()

        body_text = clean_content(lines)
        chunks = deterministic_chunk(
            body_text,
            year=meta["year"],
            wwdc=meta["wwdc"],
            title=meta["title"],
            url=meta["url"],
        )

        result = {
            "year": meta["year"],
            "wwdc": meta["wwdc"],
            "title": meta["title"],
            "url": meta["url"],
            "model": model,
            "chunks": chunks,
        }

        base = f"{meta['wwdc']}_{slugify(meta['title'])}.json"
        out_path = os.path.join(outdir, base)
        with open(out_path, "w", encoding="utf-8") as out:
            json.dump(result, out, indent=2, ensure_ascii=False)

        logger.info(f"Wrote: {out_path}")

    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    logger.info(f"Completed RAG prep in {mins:02d}:{secs:02d}")


if __name__ == "__main__":
    prep_rag()
