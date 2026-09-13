"""Live Gemini Vision Model client using standard library urllib (Zero dependencies)."""

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
from typing import Optional, Tuple

from src.models.base import BaseVisionModel
from src.schemas import EvalTask, ModelPrediction


class GeminiVisionModel(BaseVisionModel):
    """Integrates with Google's Gemini API (v1beta) for live multimodal inference."""

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        max_output_tokens: int = 1024,
    ):
        super().__init__(model_name=model_name)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"

    def _load_image_base64(self, path_or_url: str) -> Tuple[Optional[str], str, Optional[str]]:
        """Loads an image from local disk or URL and returns (base64_data, mime_type, error)."""
        try:
            if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
                req = urllib.request.Request(path_or_url, headers={"User-Agent": "MultimodalEvalHarness/1.0"})
                with urllib.request.urlopen(req, timeout=10) as response:
                    raw_bytes = response.read()
                    mime_type = response.headers.get_content_type() or "image/png"
            else:
                # Handle relative path from repo root
                abs_path = path_or_url
                if not os.path.isabs(abs_path):
                    # Check relative to repo root
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

    def predict(self, task: EvalTask) -> ModelPrediction:
        """Executes a multimodal request against the Gemini API."""
        if not self.api_key:
            return ModelPrediction(
                task_id=task.task_id,
                model_name=self.model_name,
                raw_response="[ERROR: Missing GEMINI_API_KEY]",
                latency_ms=0.0,
                error="GEMINI_API_KEY is not set. Export it in your environment: export GEMINI_API_KEY='your-key'",
            )

        # 1. Load image
        b64_image, mime_type, img_err = self._load_image_base64(task.image_path_or_url)
        if img_err:
            return ModelPrediction(
                task_id=task.task_id,
                model_name=self.model_name,
                raw_response=f"[ERROR: {img_err}]",
                latency_ms=0.0,
                error=img_err,
            )

        # 2. Build payload
        parts = [{"text": task.prompt}]
        if b64_image:
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": b64_image,
                }
            })

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }

        url = f"{self.endpoint}?key={self.api_key}"
        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")

        start_time = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                resp_json = json.loads(response.read().decode("utf-8"))

            # Extract generated text
            candidates = resp_json.get("candidates", [])
            if candidates and "content" in candidates[0]:
                content_parts = candidates[0]["content"].get("parts", [])
                text_response = "".join(part.get("text", "") for part in content_parts)
            else:
                text_response = "[EMPTY_RESPONSE]"

            usage = resp_json.get("usageMetadata", {})
            input_tokens = usage.get("promptTokenCount")
            output_tokens = usage.get("candidatesTokenCount")

            return ModelPrediction(
                task_id=task.task_id,
                model_name=self.model_name,
                raw_response=text_response.strip(),
                latency_ms=round(elapsed_ms, 2),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except urllib.error.HTTPError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            error_body = e.read().decode("utf-8", errors="replace")
            return ModelPrediction(
                task_id=task.task_id,
                model_name=self.model_name,
                raw_response=f"[HTTP Error {e.code}]",
                latency_ms=round(elapsed_ms, 2),
                error=f"HTTP {e.code}: {error_body[:200]}",
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
        """Text-only prompt generation for acting as an LLM Judge."""
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": self.max_output_tokens,
            },
        }

        url = f"{self.endpoint}?key={self.api_key}"
        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=30) as response:
            resp_json = json.loads(response.read().decode("utf-8"))

        candidates = resp_json.get("candidates", [])
        if candidates and "content" in candidates[0]:
            content_parts = candidates[0]["content"].get("parts", [])
            return "".join(part.get("text", "") for part in content_parts).strip()

        return ""

