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

import psycopg
from pgvector.psycopg import Vector, register_vector
from prompt_toolkit import prompt
from rich.console import Console
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="INFO: %(message)s")
logger = logging.getLogger(__name__)

env_model_id = os.getenv("MODEL_ID", "mlx-community/Qwen3-8B-4bit")
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

    def search_similar_chunks(self, query: str, top_k: int = 4) -> list[str]:
        raw = self.embedder.encode(query, normalize_embeddings=True)
        vec = Vector(raw.tolist())  # wrap in pgvector.Vector
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_content FROM rag_chunks ORDER BY embedding <-> %s LIMIT %s",
                (vec, top_k),
            )
            rows = cur.fetchall()
            logger.info(f"Retrieved {len(rows)} chunks")
            return [row[0] for row in rows]

    def predict(self, prompt: str) -> str:
        import time

        from mlx_lm import generate

        retrieved_chunks = self.search_similar_chunks(prompt)
        context = "\n\n".join(retrieved_chunks)
        augmented_prompt = f"Use the following context to answer the question.\n\n{context}\n\nQuestion: {prompt}"

        messages = [{"role": "user", "content": augmented_prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        # count input tokens
        logger.info(f"Input tokens: {len(formatted_prompt)}")
        logger.info("Generating response")
        start_time = time.time()
        response = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=formatted_prompt,
            verbose=True,
            prompt_cache=self.prompt_cache,
            max_tokens=2048,
        )
        return str(response)


# Placeholder for future implementations
# class OllamaClient(LLMClientInterface):
#     def __init__(self, ...): pass
#     def predict(self, prompt: str) -> str: pass


def chat_loop(client: LLMClientInterface):
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
        logger.info("Querying database")
        response = client.predict(prompt_text)
        logger.info(f"Response: {response}")
        console.print(f"[bold magenta]Model:[/] {response}\n")


if __name__ == "__main__":
    # select default client
    client = MlxLmClient()
    chat_loop(client)
