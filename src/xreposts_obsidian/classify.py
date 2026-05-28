from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
import requests

from .llm import JsonCache, LLMError, OllamaClient, TopicDecision, TopicSummary, cache_key, clean_llm_text
from .models import Repost
from .utils import slugify, truncate_words

DEFAULT_TOPIC_RULES: dict[str, list[str]] = {
    "3D Vision": [
        "3d", "point cloud", "depth", "pose estimation", "reconstruction", "multi-view", "multiview",
        "mesh", "lidar", "rgb-d", "scene understanding", "spatial ai", "geometry",
    ],
    "Robotics": [
        "robot", "robotics", "embodied", "manipulation", "navigation", "locomotion", "grasp",
        "planning", "sim2real", "ros", "autonomous robot", "mobile robot",
    ],
    "VLA Models": [
        "vla", "vision-language-action", "vision language action", "openvla", "robot foundation model",
        "embodied reasoning", "action model", "policy model",
    ],
    "SLAM and Localization": [
        "slam", "visual-inertial", "visual inertial", "vins", "localization", "mapping", "odometry",
        "bundle adjustment", "loop closure", "place recognition",
    ],
    "NeRF and Neural Rendering": [
        "nerf", "radiance field", "neural rendering", "gaussian splatting", "3dgs", "novel view",
    ],
    "Computer Vision": [
        "computer vision", "segmentation", "detection", "tracking", "optical flow", "image", "video",
        "opencv", "vision transformer", "vit", "diffusion", "sam", "clip",
    ],
    "Foundation Models": [
        "foundation model", "llm", "vlm", "large language model", "multimodal", "transformer",
        "pretraining", "fine-tuning", "finetuning", "rag",
    ],
    "AI Agents": [
        "agent", "agents", "tool use", "reasoning", "planning", "computer use", "browser agent",
        "workflow", "autonomous agent",
    ],
    "GPU": [
        "gpu", "gpus", "cuda", "nvidia", "h100", "h200", "b100", "b200", "blackwell", "grace blackwell",
        "tensor core", "vram", "nvlink", "inference acceleration", "training acceleration", "fp4", "fp8",
        "triton", "kernel", "kernels", "cuDNN", "tensorrt", "dgx", "gb200", "gb10",
    ],
    "Research Papers": [
        "paper", "arxiv", "benchmark", "dataset", "sota", "evaluation", "cvpr", "iccv", "eccv",
        "neurips", "iclr", "icml", "rss", "corl", "acl", "emnlp",
    ],
    "Open Source": [
        "github", "repo", "open source", "code release", "implementation", "library", "toolkit", "sdk",
    ],
    "Startups and Products": [
        "startup", "product", "launch", "funding", "demo", "beta", "waitlist", "company", "pricing",
    ],
}

UNCLASSIFIED = "Unclassified"


def load_topic_rules(path: str | None) -> dict[str, list[str]]:
    if not path:
        return DEFAULT_TOPIC_RULES
    topic_path = Path(path).expanduser()
    if not topic_path.exists():
        raise SystemExit(f"Topic config not found: {topic_path}")
    data = yaml.safe_load(topic_path.read_text(encoding="utf-8")) or {}
    topics = data.get("topics") if isinstance(data, dict) else data
    if not isinstance(topics, dict):
        raise SystemExit("Topic config must be a mapping or contain a top-level 'topics' mapping.")
    result: dict[str, list[str]] = {}
    for topic, keywords in topics.items():
        if isinstance(keywords, str):
            result[str(topic)] = [keywords]
        elif isinstance(keywords, list):
            result[str(topic)] = [str(keyword) for keyword in keywords]
    return result or DEFAULT_TOPIC_RULES


def _normalized_haystack(repost: Repost) -> str:
    parts = [
        repost.original_text,
        " ".join(repost.hashtags),
        " ".join(repost.mentions),
        " ".join(repost.domains),
        " ".join(repost.context_labels),
        repost.original_author.username,
        repost.original_author.name,
        repost.original_author.description,
    ]
    return "\n".join(part for part in parts if part).lower()


def topic_scores(repost: Repost, rules: dict[str, list[str]]) -> Counter[str]:
    haystack = _normalized_haystack(repost)
    scores: Counter[str] = Counter()
    for topic, keywords in rules.items():
        for keyword in keywords:
            kw = keyword.lower().strip()
            if not kw:
                continue
            if " " in kw or "-" in kw:
                if kw in haystack:
                    scores[topic] += 3
            else:
                matches = re.findall(rf"(?<![a-z0-9_]){re.escape(kw)}(?![a-z0-9_])", haystack)
                scores[topic] += len(matches)
    return scores


def infer_topics(repost: Repost, rules: dict[str, list[str]], min_score: int = 1, max_topics: int = 8) -> list[str]:
    scores = topic_scores(repost, rules)
    ranked = [topic for topic, score in scores.most_common() if score >= min_score]
    return ranked[:max_topics]


def summarize_interests(reposts: list[Repost], rules: dict[str, list[str]], min_score: int = 1) -> Counter[str]:
    counts: Counter[str] = Counter()
    for repost in reposts:
        topics = infer_topics(repost, rules, min_score=min_score)
        if topics:
            counts.update(topics)
        else:
            counts[UNCLASSIFIED] += 1
    return counts


def normalize_topic_name(value: str, known_topics: list[str] | None = None) -> str:
    raw = clean_llm_text(value, max_len=80)
    if not raw:
        return UNCLASSIFIED
    if raw.lower().strip() in {"unknown", "none", "n/a", "unclear", "unclassified"}:
        return UNCLASSIFIED
    raw = raw.replace("&", "and")
    raw = re.sub(r"[^A-Za-z0-9 +\-/]", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw:
        return UNCLASSIFIED

    known_topics = known_topics or []
    by_slug = {slugify(topic): topic for topic in known_topics + [UNCLASSIFIED]}
    match = by_slug.get(slugify(raw))
    if match:
        return match

    # Keep new topics reusable and not overly long.
    words = raw.split()
    if len(words) > 5:
        raw = " ".join(words[:5])
    small_words = {"and", "or", "of", "for", "to", "in", "on", "the", "a", "an"}
    titled = " ".join(w.lower() if w.lower() in small_words else w[:1].upper() + w[1:] for w in raw.split())
    # Preserve common acronyms.
    replacements = {
        "Ai": "AI",
        "Llms": "LLMs",
        "Llm": "LLM",
        "Vlms": "VLMs",
        "Vlm": "VLM",
        "Gpu": "GPU",
        "Gpus": "GPUs",
        "Cuda": "CUDA",
        "Nerf": "NeRF",
        "Slam": "SLAM",
        "Vla": "VLA",
    }
    for a, b in replacements.items():
        titled = re.sub(rf"\b{a}\b", b, titled)
    return titled


def fallback_title(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "Untitled repost"
    return truncate_words(text, max_words=9).strip(" .")


def fallback_summary(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return "No original text was returned by the X API."
    return truncate_words(text, max_words=32)


def fallback_decision(repost: Repost, rules: dict[str, list[str]], min_score: int = 1, source: str = "rules") -> TopicDecision:
    topics = infer_topics(repost, rules, min_score=min_score, max_topics=4)
    primary = topics[0] if topics else UNCLASSIFIED
    secondary = [topic for topic in topics[1:] if topic != primary]
    return TopicDecision(
        title=fallback_title(repost.original_text),
        summary=fallback_summary(repost.original_text),
        primary_topic=primary,
        secondary_topics=secondary,
        confidence=0.55 if topics else 0.1,
        source=source,
    )


def _classification_prompt(repost: Repost, candidate_topics: list[str]) -> list[dict[str, str]]:
    author = repost.original_author.username or repost.original_author.name or "unknown"
    payload = {
        "tweet_text": repost.original_text,
        "author": author,
        "hashtags": repost.hashtags,
        "domains": repost.domains,
        "context_labels": repost.context_labels,
        "candidate_topics": candidate_topics,
    }
    system = (
        "You classify reposted X/Twitter posts for a private Obsidian interest graph. "
        "Return STRICT JSON only. Do not include markdown. "
        "Prefer an existing candidate topic when it is a reasonable fit. "
        "Create a new concise reusable topic only when no candidate topic fits well. "
        "Use Unclassified only when the text is too vague, broken, or has no clear subject. "
        "Titles must be short, file-name friendly, and contain no square brackets or slashes. "
        "Summaries must be one sentence. "
        "GPU-related content includes NVIDIA, CUDA, kernels, tensor cores, VRAM, DGX, FP4/FP8, inference/training acceleration, and accelerator hardware."
    )
    user = (
        "Classify this repost. Return a JSON object with exactly these keys:\n"
        "{\n"
        '  "title": "5-10 word title summarizing the tweet",\n'
        '  "summary": "one sentence summary, max 35 words",\n'
        '  "primary_topic": "one topic; existing candidate preferred; new topic allowed; or Unclassified",\n'
        '  "secondary_topics": ["0-3 additional reusable topics"],\n'
        '  "confidence": 0.0,\n'
        '  "rationale": "short reason, max 18 words"\n'
        "}\n\n"
        f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def classify_repost(
    repost: Repost,
    rules: dict[str, list[str]],
    llm: OllamaClient | None,
    cache: JsonCache | None,
    min_score: int = 1,
    allow_new_topics: bool = True,
) -> TopicDecision:
    known_topics = list(rules.keys()) + [UNCLASSIFIED]
    if llm is None:
        return fallback_decision(repost, rules, min_score=min_score, source="rules")

    payload = {
        "id": repost.original_id,
        "text": repost.original_text,
        "author": repost.original_author.username,
        "hashtags": repost.hashtags,
        "domains": repost.domains,
        "context_labels": repost.context_labels,
        "known_topics": known_topics,
    }
    key = cache_key("classify_repost_v2", llm.model, payload)
    cached = cache.get(key) if cache else None
    if isinstance(cached, dict):
        return TopicDecision(**cached)

    try:
        data = llm.chat_json(_classification_prompt(repost, known_topics), temperature=0.0, num_ctx=8192)
        title = clean_llm_text(data.get("title"), max_len=90) or fallback_title(repost.original_text)
        summary = clean_llm_text(data.get("summary"), max_len=280) or fallback_summary(repost.original_text)
        primary = normalize_topic_name(str(data.get("primary_topic") or ""), known_topics)
        if primary not in known_topics and not allow_new_topics:
            primary = fallback_decision(repost, rules, min_score=min_score).primary_topic
        secondary_raw = data.get("secondary_topics") or []
        if not isinstance(secondary_raw, list):
            secondary_raw = []
        secondary: list[str] = []
        for item in secondary_raw[:4]:
            topic = normalize_topic_name(str(item), known_topics + [primary])
            if topic != UNCLASSIFIED and topic != primary and topic not in secondary:
                if topic in known_topics or allow_new_topics:
                    secondary.append(topic)
        try:
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        # Limit Unclassified. If the LLM is not confident but keyword rules give a topic, use the rules.
        if primary == UNCLASSIFIED:
            fallback = fallback_decision(repost, rules, min_score=min_score)
            if fallback.primary_topic != UNCLASSIFIED:
                primary = fallback.primary_topic
                secondary = [t for t in [*secondary, *fallback.secondary_topics] if t != primary]
                confidence = max(confidence, fallback.confidence)
        decision = TopicDecision(
            title=title,
            summary=summary,
            primary_topic=primary,
            secondary_topics=secondary[:3],
            confidence=confidence,
            source="llm",
            raw_topic=str(data.get("primary_topic") or ""),
            rationale=clean_llm_text(data.get("rationale"), max_len=120),
        )
    except (requests.RequestException, LLMError, ValueError, TypeError, json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
        decision = fallback_decision(repost, rules, min_score=min_score, source=f"rules_after_llm_error: {type(exc).__name__}")

    if cache:
        cache.set(key, decision.to_dict())
    return decision


def _topic_summary_prompt(topic: str, reposts_payload: list[dict[str, Any]]) -> list[dict[str, str]]:
    system = (
        "You synthesize private knowledge-management summaries from a user's reposted X posts. "
        "Return STRICT JSON only. Be specific, but do not invent facts not grounded in the provided texts."
    )
    user = (
        f"Create a topic summary for the topic: {topic}\n"
        "Return a JSON object with exactly these keys:\n"
        "{\n"
        '  "summary": "4-7 sentence synthesis of what these reposts collectively indicate",\n'
        '  "subthemes": ["5-10 recurring subthemes"],\n'
        '  "why_it_matters": "1-3 sentence explanation of why this cluster matters",\n'
        '  "representative_points": ["3-7 concise recurring claims or insights"]\n'
        "}\n\n"
        f"Reposts JSON:\n{json.dumps(reposts_payload, ensure_ascii=False)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def summarize_topic_with_llm(topic: str, reposts: list[tuple[Repost, TopicDecision]], llm: OllamaClient | None, cache: JsonCache | None) -> TopicSummary:
    if not reposts:
        return TopicSummary(topic=topic, summary="No reposts were assigned to this topic.", source="empty")
    samples = []
    # Give the LLM enough diversity while keeping the prompt bounded.
    for repost, decision in reposts[:80]:
        text = re.sub(r"\s+", " ", repost.original_text or "").strip()
        samples.append({
            "title": decision.title,
            "summary": decision.summary,
            "author": repost.original_author.username or repost.original_author.name,
            "text": text[:900],
            "domains": repost.domains[:5],
            "hashtags": repost.hashtags[:10],
        })
    if llm is None:
        return TopicSummary(
            topic=topic,
            summary=f"This topic contains {len(reposts)} reposts. LLM summarization was disabled, so open the related repost list below to inspect the cluster manually.",
            subthemes=[],
            why_it_matters="",
            representative_points=[],
            source="rules",
        )

    key = cache_key("topic_summary_v2", llm.model, {"topic": topic, "samples": samples})
    cached = cache.get(key) if cache else None
    if isinstance(cached, dict):
        return TopicSummary(**cached)

    try:
        data = llm.chat_json(_topic_summary_prompt(topic, samples), temperature=0.15, num_ctx=16384)
        summary = clean_llm_text(data.get("summary"), max_len=1800)
        subthemes = data.get("subthemes") if isinstance(data.get("subthemes"), list) else []
        points = data.get("representative_points") if isinstance(data.get("representative_points"), list) else []
        topic_summary = TopicSummary(
            topic=topic,
            summary=summary or f"This topic contains {len(reposts)} reposts.",
            subthemes=[clean_llm_text(item, max_len=120) for item in subthemes if clean_llm_text(item, max_len=120)][:10],
            why_it_matters=clean_llm_text(data.get("why_it_matters"), max_len=800),
            representative_points=[clean_llm_text(item, max_len=180) for item in points if clean_llm_text(item, max_len=180)][:8],
            source="llm",
        )
    except Exception as exc:  # noqa: BLE001
        topic_summary = TopicSummary(
            topic=topic,
            summary=f"This topic contains {len(reposts)} reposts. LLM summarization failed with {type(exc).__name__}; the related reposts are still linked below.",
            source=f"llm_error: {type(exc).__name__}",
        )

    if cache:
        cache.set(key, topic_summary.to_dict())
    return topic_summary
