#!/usr/bin/env python3
"""
Simple CLI chat client supporting swappable LLM backends.
Default: MLX-LM (via mlx-lm Python SDK)

Other options (Ollama, LM Studio) can be plugged in by
implementing the LLMClientInterface.
"""

import abc
import logging
import os
import sys
import time

import psycopg
from mlx_lm import generate
from pgvector.psycopg import Vector, register_vector
from prompt_toolkit import prompt
from rich.console import Console
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")
logger = logging.getLogger(__name__)

env_model_id = os.getenv("MODEL_ID", "mlx-community/Qwen3-4B-4bit")
env_context = int(os.getenv("CONTEXT_LENGTH", "32768"))


class LLMClientInterface(abc.ABC):
    """Abstract interface for LLM backends."""

    @abc.abstractmethod
    def predict(self, prompt: str) -> tuple[str, float]:
        """Send prompt to model and return its response and tokens/sec."""
        pass


class MlxLmClient(LLMClientInterface):
    """Concrete LLMClient using the mlx-lm SDK."""

    def __init__(
        self,
        model_id: str = env_model_id,
        context_length: int = env_context,
        db_url: str = "postgresql://admin:admin@localhost:5432/wwdc_vector",
        table_name: str = "rag_chunks",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        try:
            from mlx_lm.models.cache import make_prompt_cache
            from mlx_lm.utils import load
        except ImportError:
            print(
                "Error: mlx-lm package not installed.\nInstall with 'pip install mlx-lm'",
                file=sys.stderr,
            )
            sys.exit(1)

        self.model, self.tokenizer = load(model_id)
        self.prompt_cache = make_prompt_cache(self.model)

        self.embedder = SentenceTransformer(embedding_model)
        self.table_name = table_name
        self.db_url = db_url
        self.conn = psycopg.connect(self.db_url)
        register_vector(self.conn)

    def search_similar_chunks(self, query: str, top_k: int = 4) -> list[dict]:
        raw = self.embedder.encode(query, normalize_embeddings=True)
        vec = Vector(raw.tolist())  # wrap in pgvector.Vector
        with self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT chunk_content, chunk_title, chunk_summary, doc_title, doc_url, doc_year
                FROM {self.table_name}
                ORDER BY embedding <-> %s
                LIMIT %s
                """,
                (vec, top_k),
            )
            rows = cur.fetchall()
            logger.info(f"Retrieved {len(rows)} chunks")
            return [
                {
                    "content": row[0],
                    "title": row[1],
                    "summary": row[2],
                    "doc_title": row[3],
                    "doc_url": row[4],
                    "doc_year": row[5],
                }
                for row in rows
            ]

    def predict(self, prompt: str, show_think: bool = False) -> str:
        retrieved_chunks = self.search_similar_chunks(prompt)
        context_parts = []
        citations = []

        for chunk in retrieved_chunks:
            section_title = f"### {chunk['title']}\n" if chunk["title"] else ""
            summary_text = f"{chunk['summary']}\n\n" if chunk["summary"] else ""
            context_parts.append(f"{section_title}{summary_text}{chunk['content']}")
            citations.append(
                f"- [{chunk['doc_title']} (WWDC {chunk['doc_year']})]({chunk['doc_url']})"
            )

        context = "\n\n".join(context_parts)
        citation_text = "\n\nSources:\n" + "\n".join(citations)

        augmented_prompt = f"Use the following context to answer the question.\n\n{context}\n\nQuestion: {prompt}"

        system_prompt = (
            "You are a helpful assistant that answers user questions about Apple’s WWDC "
            "(Worldwide Developers Conference) events using a collection of documents stored in a "
            "vector database. These documents are based on official WWDC transcripts and summaries "
            "from past conferences. Your job is to retrieve relevant content and generate clear, "
            "informative responses.\n\n"
            "Your behavior:\n"
            "- Always ground your answers in the retrieved context.\n"
            "- Be concise, technically accurate, and helpful to developers.\n"
            "- If relevant, cite the source by linking to the original WWDC session so the user can "
            "“learn more.”\n"
            "- Do not fabricate information. If the information is not available in the context, say so.\n"
            "- You are friendly and professional but avoid unnecessary chit-chat.\n\n"
            "Citation format:\n"
            "> Learn more: [Session Title](https://developer.apple.com/videos/play/wwdcYYYY/###/)"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_prompt},
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        logger.info(f"Input tokens: {len(formatted_prompt)}")
        logger.info("Generating response")
        start_time = time.time()
        response = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=formatted_prompt,
            verbose=False,
            prompt_cache=self.prompt_cache,
            max_tokens=2048,
        )
        response_str = str(response)
        if not show_think:
            import re

            response_str = re.sub(
                r"<think>.*?</think>", "", response_str, flags=re.DOTALL
            )
        return f"{response_str.strip()}\n\n{citation_text}"


# Placeholder for future implementations
# class OllamaClient(LLMClientInterface):
#     def __init__(self, ...): pass
#     def predict(self, prompt: str) -> str: pass


def chat_loop(client: LLMClientInterface, show_think: bool = False):
    console = Console()
    console.print("[bold cyan]MLX-LM Chat Client[/] (type 'exit' or 'quit' to stop)")

    while True:
        try:
            prompt_text = prompt("You: ")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red]Exiting.[/]")
            break
        if not prompt_text or prompt_text.lower() in ("exit", "quit"):
            console.print("[bold green]Goodbye.[/]")
            break
        start_time = time.time()
        logger.info("Querying database")
        response = client.predict(prompt_text, show_think=show_think)
        logger.info(f"Response: {response}")
        console.print(f"[bold magenta]Model:[/] {response}\n")
        elapsed = time.time() - start_time
        console.print(
            f"[grey62]⏱️ Elapsed time: {elapsed:.2f} seconds[/]", style="grey62"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-think",
        action="store_true",
        help="Show &lt;think&gt; sections in the response",
    )
    args = parser.parse_args()

    client = MlxLmClient()
    chat_loop(client, show_think=args.show_think)
