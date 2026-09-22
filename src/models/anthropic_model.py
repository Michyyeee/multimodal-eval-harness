"""Live Anthropic Claude Vision Model client using standard library urllib (Zero dependencies)."""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from src.models.base import BaseVisionModel
from src.models.utils import calculate_token_cost, load_image_base64
from src.schemas import EvalTask, ModelPrediction


class AnthropicVisionModel(BaseVisionModel):
    """Integrates with Anthropic's Messages API for multimodal VLM inference (e.g. Claude 3.5 Sonnet)."""

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        api_base: str = "https://api.anthropic.com/v1",
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
    ):
        super().__init__(model_name=model_name)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.api_base = api_base.rstrip("/")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.endpoint = f"{self.api_base}/messages"

    def predict(self, task: EvalTask) -> ModelPrediction:
        """Executes a multimodal request against the Anthropic Messages API."""
        if not self.api_key:
            return ModelPrediction(
                task_id=task.task_id,
                model_name=self.model_name,
                raw_response="[ERROR: Missing ANTHROPIC_API_KEY]",
                latency_ms=0.0,
                error="ANTHROPIC_API_KEY is not set. Export it in your environment: export ANTHROPIC_API_KEY='sk-ant-...'",
            )

        # 1. Load image
        b64_image, mime_type, img_err = load_image_base64(task.image_path_or_url)
        if img_err:
            return ModelPrediction(
                task_id=task.task_id,
                model_name=self.model_name,
                raw_response=f"[ERROR: {img_err}]",
                latency_ms=0.0,
                error=img_err,
            )

        # 2. Build Anthropic multimodal content blocks
        content_blocks = []
        if b64_image:
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": b64_image,
                },
            })
        content_blocks.append({"type": "text", "text": task.prompt})

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content_blocks}],
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "User-Agent": "MultimodalEvalHarness/1.0",
        }
        req = urllib.request.Request(self.endpoint, data=data_bytes, headers=headers, method="POST")

        start_time = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                resp_json = json.loads(response.read().decode("utf-8"))

            content_list = resp_json.get("content", [])
            text_response = "".join(item.get("text", "") for item in content_list if item.get("type") == "text")
            if not text_response:
                text_response = "[EMPTY_RESPONSE]"

            usage = resp_json.get("usage", {})
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            cost_usd = calculate_token_cost(self.model_name, input_tokens, output_tokens)

            return ModelPrediction(
                task_id=task.task_id,
                model_name=self.model_name,
                raw_response=text_response.strip(),
                latency_ms=round(elapsed_ms, 2),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )

        except urllib.error.HTTPError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            error_body = e.read().decode("utf-8", errors="replace")
            return ModelPrediction(
                task_id=task.task_id,
                model_name=self.model_name,
                raw_response=f"[HTTP Error {e.code}]",
                latency_ms=round(elapsed_ms, 2),
                error=f"HTTP {e.code}: {error_body[:250]}",
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ModelPrediction(
                task_id=task.task_id,
                model_name=self.model_name,
                raw_response="[Network/Connection Error]",
                latency_ms=round(elapsed_ms, 2),
                error=str(e),
            )

    def generate(self, prompt: str) -> str:
        """Text-only generation for acting as an LLM Judge."""
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured.")

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_output_tokens,
            "temperature": 0.0,
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        req = urllib.request.Request(self.endpoint, data=data_bytes, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=45) as response:
            resp_json = json.loads(response.read().decode("utf-8"))

        content_list = resp_json.get("content", [])
        return "".join(item.get("text", "") for item in content_list if item.get("type") == "text").strip()
