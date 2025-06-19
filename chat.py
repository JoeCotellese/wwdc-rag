#!/usr/bin/env python3
"""
Simple CLI chat client supporting swappable LLM backends.
Default: MLX-LM (via mlx-lm Python SDK)

Other options (Ollama, LM Studio) can be plugged in by
implementing the LLMClientInterface.
"""

import abc
import os
import sys

from prompt_toolkit import prompt
from rich.console import Console

env_model_id = os.getenv("MODEL_ID", "mlx-community/Qwen3-8B-4bit")
env_context = int(os.getenv("CONTEXT_LENGTH", "2048"))


class LLMClientInterface(abc.ABC):
    """Abstract interface for LLM backends."""

    @abc.abstractmethod
    def predict(self, prompt: str) -> str:
        """Send prompt to model and return its response."""
        pass


class MlxLmClient(LLMClientInterface):
    """Concrete LLMClient using the mlx-lm SDK."""

    def __init__(
        self,
        model_id: str = env_model_id,
        context_length: int = env_context,
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

    def predict(self, prompt: str) -> str:
        from mlx_lm import generate

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        response = generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=formatted_prompt,
            verbose=True,
            prompt_cache=self.prompt_cache,
        )
        return response if isinstance(response, str) else str(response)


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
        response = client.predict(prompt_text)
        console.print(f"[bold magenta]Model:[/] {response}\n")


if __name__ == "__main__":
    # select default client
    client = MlxLmClient()
    chat_loop(client)
