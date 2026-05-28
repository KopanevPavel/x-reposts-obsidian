from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests

from .utils import ensure_parent


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_LLM_MODEL = "qwen3:32b"


class LLMError(RuntimeError):
    pass


@dataclass
class TopicDecision:
    title: str
    summary: str
    primary_topic: str
    secondary_topics: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "unknown"
    raw_topic: str = ""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TopicSummary:
    topic: str
    summary: str
    subthemes: list[str] = field(default_factory=list)
    why_it_matters: str = ""
    representative_points: list[str] = field(default_factory=list)
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JsonCache:
    def __init__(self, path: str | Path | None):
        self.path = Path(path).expanduser().resolve() if path else None
        self.data: dict[str, Any] = {}
        self._lock = threading.RLock()
        if self.path and self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data = loaded
            except Exception:
                self.data = {}

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self.data[key] = value

    def save(self, path: str | Path | None = None) -> None:
        out = Path(path).expanduser().resolve() if path else self.path
        if not out:
            return
        ensure_parent(out)
        with self._lock:
            snapshot = dict(self.data)
        out.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def cache_key(kind: str, model: str, payload: Any) -> str:
    blob = json.dumps({"kind": kind, "model": model, "payload": payload}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class OllamaClient:
    def __init__(self, base_url: str = DEFAULT_OLLAMA_BASE_URL, model: str = DEFAULT_LLM_MODEL, timeout_seconds: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def list_models(self) -> list[str]:
        response = requests.get(f"{self.base_url}/api/tags", timeout=10)
        response.raise_for_status()
        data = response.json()
        models = data.get("models") if isinstance(data, dict) else []
        result = []
        for item in models or []:
            if isinstance(item, dict) and item.get("name"):
                result.append(str(item["name"]))
        return result

    def check(self) -> dict[str, Any]:
        models = self.list_models()
        return {
            "base_url": self.base_url,
            "model": self.model,
            "available_models": models,
            "model_available": self.model in models or any(name.split(":", 1)[0] == self.model.split(":", 1)[0] for name in models),
        }

    def chat_json(self, messages: list[dict[str, str]], temperature: float = 0.1, num_ctx: int = 8192) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }
        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_seconds)
        if response.status_code >= 400:
            raise LLMError(f"Ollama HTTP {response.status_code}: {response.text}")
        data = response.json()
        content = ((data.get("message") or {}).get("content") or "").strip()
        parsed = extract_json_object(content)
        if not isinstance(parsed, dict):
            raise LLMError(f"Ollama did not return a JSON object. Raw content: {content[:500]}")
        return parsed


def extract_json_object(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def clean_llm_text(value: Any, max_len: int = 180) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \"'`")
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text
