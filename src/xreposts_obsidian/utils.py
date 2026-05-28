from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


def utc_now_epoch() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp())


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: object) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_filename(name: str, fallback: str = "unknown") -> str:
    cleaned = name.strip() if name else ""
    cleaned = re.sub(r"[\\/:*?\"<>|#^\n\r\t]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    if not cleaned:
        cleaned = fallback
    return cleaned[:140]


def wikilink(folder: str, display: str, note_name: str | None = None) -> str:
    target = safe_filename(note_name or display)
    if display == target:
        return f"[[{folder}/{target}]]"
    return f"[[{folder}/{target}|{display}]]"


def domain_from_url(url: str) -> str:
    try:
        domain = urlparse(url).netloc.lower().strip()
    except Exception:
        return ""
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def iso_date(value: str | None) -> str:
    if not value:
        return "undated"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
    except Exception:
        match = re.search(r"(19|20)\d{2}-\d{2}-\d{2}", value)
        if match:
            return match.group(0)
    return "undated"


def iso_datetime_for_filename(value: str | None) -> str:
    if not value:
        return "undated"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%d %H-%M")
    except Exception:
        date = iso_date(value)
        return date if date != "undated" else "undated"


def slugify(value: str, fallback: str = "unknown") -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or fallback


def truncate_words(value: str, max_words: int = 28) -> str:
    words = re.sub(r"\s+", " ", value or "").strip().split(" ")
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip() + "…"


def yaml_string(value: str) -> str:
    return json.dumps(value or "", ensure_ascii=False)


def markdown_quote(text: str) -> str:
    lines = (text or "").splitlines() or [""]
    return "\n".join(f"> {line}" if line else ">" for line in lines)
