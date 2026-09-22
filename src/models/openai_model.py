"""Live OpenAI Vision Model client using standard library urllib (Zero dependencies)."""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

from src.models.base import BaseVisionModel
from src.models.utils import calculate_token_cost, load_image_base64
from src.schemas import EvalTask, ModelPrediction


class OpenAIVisionModel(BaseVisionModel):
    """Integrates with OpenAI's Chat Completions API for multimodal VLM inference (e.g. GPT-4o)."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_base: str = "https://api.openai.com/v1",
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
    ):
        super().__init__(model_name=model_name)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base.rstrip("/")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.endpoint = f"{self.api_base}/chat/completions"

    def predict(self, task: EvalTask) -> ModelPrediction:
        """Executes a multimodal request against the OpenAI API."""
        if not self.api_key:
            return ModelPrediction(
                task_id=task.task_id,
                model_name=self.model_name,
                raw_response="[ERROR: Missing OPENAI_API_KEY]",
                latency_ms=0.0,
                error="OPENAI_API_KEY is not set. Export it in your environment: export OPENAI_API_KEY='sk-...'",
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

        # 2. Build multimodal user content
        content_parts = [{"type": "text", "text": task.prompt}]
        if b64_image:
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64_image}",
                    "detail": "auto",
                },
            })

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content_parts}],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "MultimodalEvalHarness/1.0",
        }
        req = urllib.request.Request(self.endpoint, data=data_bytes, headers=headers, method="POST")

        start_time = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=45) as response:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                resp_json = json.loads(response.read().decode("utf-8"))

            choices = resp_json.get("choices", [])
            if choices and "message" in choices[0]:
                text_response = choices[0]["message"].get("content", "")
            else:
                text_response = "[EMPTY_RESPONSE]"

            usage = resp_json.get("usage", {})
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
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
            raise ValueError("OPENAI_API_KEY is not configured.")

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": self.max_output_tokens,
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(self.endpoint, data=data_bytes, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=45) as response:
            resp_json = json.loads(response.read().decode("utf-8"))

        choices = resp_json.get("choices", [])
        if choices and "message" in choices[0]:
            return choices[0]["message"].get("content", "").strip()
        return ""
