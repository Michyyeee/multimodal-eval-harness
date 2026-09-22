"""Multimodal image utilities and cost estimation across model providers (Zero external dependencies)."""

import base64
import mimetypes
import os
import urllib.error
import urllib.request
from typing import Dict, Optional, Tuple


def load_image_base64(path_or_url: str) -> Tuple[Optional[str], str, Optional[str]]:
    """Loads an image from local disk or URL and returns (base64_data, mime_type, error)."""
    try:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            req = urllib.request.Request(path_or_url, headers={"User-Agent": "MultimodalEvalHarness/1.0"})
            with urllib.request.urlopen(req, timeout=15) as response:
                raw_bytes = response.read()
                mime_type = response.headers.get_content_type() or "image/png"
        else:
            abs_path = path_or_url
            if not os.path.isabs(abs_path):
                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                candidate_path = os.path.join(repo_root, path_or_url)
                if os.path.exists(candidate_path):
                    abs_path = candidate_path

            if not os.path.exists(abs_path):
                return None, "image/png", f"Image file not found: {path_or_url}"

            with open(abs_path, "rb") as f:
                raw_bytes = f.read()

            guessed_mime, _ = mimetypes.guess_type(abs_path)
            mime_type = guessed_mime or "image/png"

        b64_encoded = base64.b64encode(raw_bytes).decode("utf-8")
        return b64_encoded, mime_type, None

    except Exception as e:
        return None, "image/png", f"Failed to load image: {str(e)}"


# Standard pricing table (USD per 1,000,000 tokens)
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    # model_prefix_or_name: (input_price_per_m, output_price_per_m)
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o1": (15.00, 60.00),
    "o3-mini": (1.10, 4.40),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    "mock": (0.0, 0.0),
}


def calculate_token_cost(model_name: str, input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    """Calculates approximate USD cost for a model call."""
    if input_tokens is None or output_tokens is None:
        return None

    norm_name = model_name.lower().replace("openai:", "").replace("anthropic:", "").replace("gemini:", "")
    price = None
    for prefix, p in MODEL_PRICING.items():
        if prefix in norm_name:
            price = p
            break

    if not price:
        return None

    in_cost = (input_tokens / 1_000_000.0) * price[0]
    out_cost = (output_tokens / 1_000_000.0) * price[1]
    return round(in_cost + out_cost, 6)
