import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


def get_available_prompts(model_name: str) -> dict[str, Any]:
    """Read prompt config from the Hub without downloading model weights."""
    config_path = hf_hub_download(
        repo_id=model_name,
        filename="config_sentence_transformers.json",
    )
    with Path(config_path).open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    prompts = cfg.get("prompts", {}) or {}
    return prompts
