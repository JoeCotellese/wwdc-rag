import logging
from typing import List, Protocol

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
