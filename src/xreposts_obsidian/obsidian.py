from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is an optional runtime nicety
    tqdm = None

_T = TypeVar("_T")

from .classify import UNCLASSIFIED, classify_repost, summarize_topic_with_llm
from .llm import JsonCache, OllamaClient, TopicDecision, TopicSummary
from .models import Repost
from .utils import iso_datetime_for_filename, markdown_quote, safe_filename, slugify, wikilink, write_json, write_text


@dataclass
class NoteRef:
    repost_id: str
    original_id: str
    target: str
    filename: str
    display: str
    primary_topic: str
    topics: list[str]
    author: str
    reposted_at: str
    summary: str
    decision_source: str


PALETTE_HEX = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
    "#d35400", "#16a085", "#8e44ad", "#2980b9", "#27ae60", "#c0392b", "#7f8c8d", "#f1c40f",
    "#ff6b6b", "#4dabf7", "#51cf66", "#ffd43b", "#cc5de8", "#22b8cf", "#ffa94d", "#868e96",
]


def _progress(
    iterable: Iterable[_T],
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    enabled: bool = True,
) -> Iterable[_T]:
    """Return a tqdm progress iterator when available, otherwise the original iterable."""
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True)

def _progress_write(message: str, *, enabled: bool = True) -> None:
    if enabled and tqdm is not None:
        tqdm.write(message)
    else:
        print(message)


def _hex_to_rgb_int(hex_color: str) -> int:
    value = hex_color.strip().lstrip("#")
    return int(value, 16)


def _author_display(repost: Repost) -> str:
    username = repost.original_author.username
    if username:
        return "@" + username.lstrip("@")
    return "Unknown"


def _author_note_name(repost: Repost) -> str:
    username = repost.original_author.username
    if username:
        return "@" + username.lstrip("@")
    return "Unknown author"


def _list_links(folder: str, values: list[str]) -> str:
    if not values:
        return "None"
    return ", ".join(wikilink(folder, value) for value in values)


def _metric_lines(metrics: dict) -> str:
    if not metrics:
        return "No public metrics returned by API."
    order = ["retweet_count", "reply_count", "like_count", "quote_count", "bookmark_count", "impression_count"]
    lines = []
    for key in order:
        if key in metrics:
            lines.append(f"- {key}: {metrics[key]}")
    for key, value in sorted(metrics.items()):
        if key not in order:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _unique_stem(base: str, used: set[str]) -> str:
    stem = safe_filename(base, fallback="untitled repost")
    if stem not in used:
        used.add(stem)
        return stem
    for i in range(2, 1000):
        candidate = safe_filename(f"{stem} {i}")
        if candidate not in used:
            used.add(candidate)
            return candidate
    candidate = safe_filename(f"{stem} {len(used) + 1}")
    used.add(candidate)
    return candidate


def _repost_stem(repost: Repost, decision: TopicDecision, used: set[str]) -> str:
    title = safe_filename(decision.title, fallback="Untitled repost")
    time_part = iso_datetime_for_filename(repost.reposted_at)
    author = _author_note_name(repost)
    # Requested format: title(summary via LLM) [time] [author].
    return _unique_stem(f"{title} [{time_part}] [{author}]", used)


def _topic_cluster_tag(topic: str) -> str:
    return f"x/cluster/{slugify(topic)}"


def _topic_note_tag(topic: str) -> str:
    return f"x/topic/{slugify(topic)}"


def _write_repost_note(vault: Path, repost: Repost, decision: TopicDecision, note_target: str) -> None:
    author_display = _author_display(repost)
    author_link = wikilink("Authors", author_display, _author_note_name(repost))
    all_topics = [decision.primary_topic] + [t for t in decision.secondary_topics if t != decision.primary_topic]
    topic_links = _list_links("Topics", all_topics if all_topics else [UNCLASSIFIED])
    domain_links = _list_links("Domains", repost.domains)
    hashtag_links = _list_links("Hashtags", repost.hashtags)
    mention_links = _list_links("Authors", repost.mentions)
    url_lines = "\n".join(f"- {url}" for url in repost.urls) if repost.urls else "- No expanded URLs returned by API."
    context_lines = "\n".join(f"- {label}" for label in repost.context_labels) if repost.context_labels else "- No context labels returned by API."
    cluster_tag = _topic_cluster_tag(decision.primary_topic)
    secondary_tags = [f"x/secondary-topic/{slugify(t)}" for t in decision.secondary_topics]
    tag_lines = ["  - x/repost", f"  - {cluster_tag}", *[f"  - {tag}" for tag in secondary_tags]]

    content = f"""---
type: x-repost
repost_id: {json.dumps(repost.repost_id)}
original_id: {json.dumps(repost.original_id)}
reposted_at: {json.dumps(repost.reposted_at)}
original_created_at: {json.dumps(repost.original_created_at)}
original_author: {json.dumps(author_display)}
original_url: {json.dumps(repost.original_url)}
primary_topic: {json.dumps(decision.primary_topic)}
topics: {json.dumps(all_topics, ensure_ascii=False)}
llm_title: {json.dumps(decision.title, ensure_ascii=False)}
llm_summary: {json.dumps(decision.summary, ensure_ascii=False)}
llm_confidence: {decision.confidence}
classification_source: {json.dumps(decision.source)}
tags:
{chr(10).join(tag_lines)}
---

# {decision.title}

**Summary:** {decision.summary}  
**Primary topic:** {wikilink("Topics", decision.primary_topic)}  
**Other topics:** {_list_links("Topics", decision.secondary_topics)}  
**Reposted at:** {repost.reposted_at or "Unknown"}  
**Original post:** {repost.original_url}  
**Original author:** {author_link}  
**Domains:** {domain_links}  
**Hashtags:** {hashtag_links}  
**Mentions:** {mention_links}  
**Classifier:** {decision.source}, confidence {decision.confidence:.2f}

## Original text

{markdown_quote(repost.original_text)}

## LLM rationale

{decision.rationale or "No rationale returned."}

## Links

{url_lines}

## X context labels

{context_lines}

## Public metrics on original post

{_metric_lines(repost.public_metrics)}
"""
    write_text(vault / f"{note_target}.md", content)


def _link(target: str, display: str | None = None) -> str:
    if display and display != target:
        return f"[[{target}|{display}]]"
    return f"[[{target}]]"


def _write_author_note(vault: Path, name: str, refs: list[NoteRef]) -> None:
    lines = [f"# Author: {name}", "", f"Count: {len(refs)}", "", "## Topic distribution", ""]
    counts = Counter(ref.primary_topic for ref in refs)
    for topic, count in counts.most_common():
        lines.append(f"- [[Topics/{safe_filename(topic)}|{topic}]] — {count}")
    lines.extend(["", "## Related reposts", ""])
    for ref in sorted(refs, key=lambda r: r.reposted_at, reverse=True):
        lines.append(f"- {_link(ref.target, ref.display)} — [[Topics/{safe_filename(ref.primary_topic)}|{ref.primary_topic}]]")
    write_text(vault / "Authors" / f"{safe_filename(name)}.md", "\n".join(lines) + "\n")


def _write_simple_index_note(vault: Path, folder: str, name: str, title_prefix: str, refs: list[NoteRef]) -> None:
    lines = [f"# {title_prefix}: {name}", "", f"Count: {len(refs)}", "", "## Related reposts", ""]
    for ref in sorted(refs, key=lambda r: r.reposted_at, reverse=True):
        lines.append(f"- {_link(ref.target, ref.display)} — [[Topics/{safe_filename(ref.primary_topic)}|{ref.primary_topic}]]")
    write_text(vault / folder / f"{safe_filename(name)}.md", "\n".join(lines) + "\n")


def _write_topic_note(vault: Path, topic: str, refs: list[NoteRef], summary: TopicSummary) -> None:
    tag_lines = ["  - x/topic", f"  - {_topic_note_tag(topic)}", f"  - {_topic_cluster_tag(topic)}"]
    subthemes = "\n".join(f"- {item}" for item in summary.subthemes) if summary.subthemes else "- No subthemes generated."
    points = "\n".join(f"- {item}" for item in summary.representative_points) if summary.representative_points else "- No representative points generated."
    topic_refs = sorted(refs, key=lambda r: r.reposted_at, reverse=True)
    repost_lines = "\n".join(f"- {_link(ref.target, ref.display)} — {ref.author}" for ref in topic_refs)
    author_counts = Counter(ref.author for ref in refs)
    author_lines = "\n".join(f"- [[Authors/{safe_filename(author)}|{author}]] — {count}" for author, count in author_counts.most_common(20))

    content = f"""---
type: x-topic
topic: {json.dumps(topic, ensure_ascii=False)}
count: {len(refs)}
tags:
{chr(10).join(tag_lines)}
---

# Topic: {topic}

Count: {len(refs)}  
Graph cluster tag: `#{_topic_cluster_tag(topic)}`  
Summary source: {summary.source}

## Major summary

{summary.summary}

## Recurring subthemes

{subthemes}

## Why it matters

{summary.why_it_matters or "No separate explanation generated."}

## Representative points

{points}

## Top authors in this topic

{author_lines or "No author data."}

## Related reposts

{repost_lines or "No reposts."}
"""
    write_text(vault / "Topics" / f"{safe_filename(topic)}.md", content)


def _write_graph_config(vault: Path, topics: list[str]) -> None:
    color_groups = []
    for i, topic in enumerate(topics):
        color = PALETTE_HEX[i % len(PALETTE_HEX)]
        color_groups.append({
            "query": f"tag:#x/cluster/{slugify(topic)}",
            "color": {"a": 1, "rgb": _hex_to_rgb_int(color)},
        })
    graph = {
        "collapse-filter": False,
        "search": "",
        "showTags": False,
        "showAttachments": False,
        "hideUnresolved": False,
        "showOrphans": True,
        "collapse-color-groups": False,
        "colorGroups": color_groups,
        "collapse-display": False,
        "showArrow": False,
        "textFadeMultiplier": 0,
        "nodeSizeMultiplier": 1,
        "lineSizeMultiplier": 1,
        "collapse-forces": False,
        "centerStrength": 0.518713248970312,
        "repelStrength": 10,
        "linkStrength": 1,
        "linkDistance": 250,
        "scale": 1,
        "close": False,
    }
    write_json(vault / ".obsidian" / "graph.json", graph)


def _write_graph_guide(vault: Path, primary_topics: list[str]) -> None:
    top = primary_topics[:10]
    lines = [
        "# Graph Guide",
        "",
        "This vault uses one cluster tag per primary topic. The topic note and all repost notes assigned to that primary topic share the same `#x/cluster/...` tag, so Obsidian graph color groups can color the parent topic and its children the same way.",
        "",
        "The project writes `.obsidian/graph.json` automatically. If Obsidian does not pick it up immediately, close and reopen the vault, then open Graph View → Settings → Groups.",
        "",
        "## Useful graph filters",
        "",
        "Show only repost notes:",
        "",
        "```text",
        "tag:#x/repost",
        "```",
        "",
        "Show topic notes plus repost notes:",
        "",
        "```text",
        "path:Topics OR tag:#x/repost",
        "```",
        "",
        "Show only topic nodes:",
        "",
        "```text",
        "path:Topics",
        "```",
        "",
        "Show topic clusters without authors/domains/hashtags:",
        "",
        "```text",
        "path:Topics OR path:Reposts",
        "```",
        "",
        "Show authors and domains around reposts:",
        "",
        "```text",
        "tag:#x/repost OR path:Authors OR path:Domains",
        "```",
        "",
        "## Single-cluster examples",
        "",
    ]
    for topic in top:
        tag = f"x/cluster/{slugify(topic)}"
        safe_topic = safe_filename(topic)
        lines.extend([
            f"### {topic}",
            "",
            "Topic note + child reposts in the same cluster color:",
            "",
            "```text",
            f"tag:#{tag}",
            "```",
            "",
            "Only reposts in this cluster:",
            "",
            "```text",
            f"tag:#x/repost tag:#{tag}",
            "```",
            "",
            "Topic note and primary-topic folder:",
            "",
            "```text",
            f"path:\"Topics/{safe_topic}\" OR path:\"Reposts/{safe_topic}\"",
            "```",
            "",
            f"Start note: [[Topics/{safe_topic}|{topic}]]",
            "",
        ])
    lines.extend([
        "## Local graph workflow",
        "",
        "Open any topic note, then open the local graph for that note. Depth 1 shows the topic and direct reposts. Depth 2 adds authors, domains, hashtags, and secondary topics.",
        "",
        "## Useful starting notes",
        "",
        "- [[Interest Summary]]",
        "- [[Graph Guide]]",
    ])
    for topic in top[:8]:
        lines.append(f"- [[Topics/{safe_filename(topic)}|{topic}]]")
    write_text(vault / "Graph Guide.md", "\n".join(lines) + "\n")


def _write_data_readme(vault: Path) -> None:
    content = """# Data folder

This folder is intentionally populated so the generated vault is reproducible and inspectable.

Files written here:

- `reposts.jsonl` — raw normalized repost objects returned by the project.
- `topic_assignments.jsonl` — one row per repost with the LLM/rules topic decision and generated note path.
- `topic_summaries.json` — LLM/rules summaries used in `Topics/*.md`.
- `topics.json` — counts, slugs, color tags, and graph colors for topic clusters.
- `manifest.json` — generation metadata and counts.
- `llm_cache.json` — cached LLM classification and summary outputs for this build.
- `coverage.json` — API coverage metadata written after fetching finishes.
"""
    write_text(vault / "Data" / "README.md", content)




def _classify_reposts(
    reposts: list[Repost],
    topic_rules: dict[str, list[str]],
    min_topic_score: int,
    llm: OllamaClient | None,
    cache: JsonCache | None,
    allow_new_topics: bool,
    llm_concurrency: int = 1,
    show_progress: bool = True,
) -> list[TopicDecision]:
    """Classify reposts, optionally issuing multiple independent Ollama calls in parallel.

    The returned decisions preserve the same order as ``reposts`` so note names and
    downstream graph generation remain deterministic.
    """
    if not reposts:
        return []

    concurrency = max(1, int(llm_concurrency or 1))
    if llm is None or concurrency == 1 or len(reposts) == 1:
        decisions: list[TopicDecision] = []
        label = "LLM classify" if llm else "Rule classify"
        iterator = _progress(enumerate(reposts, start=1), total=len(reposts), desc=label, unit="repost", enabled=show_progress)
        for idx, repost in iterator:
            decisions.append(
                classify_repost(
                    repost=repost,
                    rules=topic_rules,
                    llm=llm,
                    cache=cache,
                    min_score=min_topic_score,
                    allow_new_topics=allow_new_topics,
                )
            )
            if idx % 25 == 0 and cache:
                cache.save()
        return decisions

    concurrency = min(concurrency, len(reposts))
    _progress_write(f"Classifying {len(reposts)} reposts with LLM concurrency={concurrency}...", enabled=show_progress)

    decisions: list[TopicDecision | None] = [None] * len(reposts)
    completed = 0

    def classify_one(index: int, repost: Repost) -> tuple[int, TopicDecision]:
        decision = classify_repost(
            repost=repost,
            rules=topic_rules,
            llm=llm,
            cache=cache,
            min_score=min_topic_score,
            allow_new_topics=allow_new_topics,
        )
        return index, decision

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(classify_one, index, repost) for index, repost in enumerate(reposts)]
        iterator = _progress(as_completed(futures), total=len(futures), desc="LLM classify", unit="repost", enabled=show_progress)
        for future in iterator:
            index, decision = future.result()
            decisions[index] = decision
            completed += 1
            if completed == len(reposts) or completed % 25 == 0:
                if cache:
                    cache.save()

    missing = [i for i, decision in enumerate(decisions) if decision is None]
    if missing:
        raise RuntimeError(f"Classification finished with missing decisions at indexes: {missing[:10]}")
    return [decision for decision in decisions if decision is not None]

def build_vault(
    reposts: list[Repost],
    vault_path: str | Path,
    topic_rules: dict[str, list[str]],
    min_topic_score: int = 1,
    clear: bool = False,
    llm: OllamaClient | None = None,
    llm_cache_path: str | Path | None = None,
    allow_new_topics: bool = True,
    llm_concurrency: int = 1,
    show_progress: bool = True,
) -> dict[str, int]:
    vault = Path(vault_path).expanduser().resolve()
    if clear and vault.exists():
        shutil.rmtree(vault)
    vault.mkdir(parents=True, exist_ok=True)

    for folder in ["Reposts", "Authors", "Topics", "Domains", "Hashtags", "Data", ".obsidian"]:
        (vault / folder).mkdir(parents=True, exist_ok=True)
    _write_data_readme(vault)

    cache_path = Path(llm_cache_path).expanduser().resolve() if llm_cache_path else vault / "Data" / "llm_cache.json"
    cache = JsonCache(cache_path)

    assignments: list[tuple[Repost, TopicDecision, NoteRef]] = []
    by_topic: dict[str, list[NoteRef]] = defaultdict(list)
    by_primary_topic: dict[str, list[tuple[Repost, TopicDecision]]] = defaultdict(list)
    by_author: dict[str, list[NoteRef]] = defaultdict(list)
    by_domain: dict[str, list[NoteRef]] = defaultdict(list)
    by_hashtag: dict[str, list[NoteRef]] = defaultdict(list)
    used_stems: set[str] = set()

    decisions = _classify_reposts(
        reposts=reposts,
        topic_rules=topic_rules,
        min_topic_score=min_topic_score,
        llm=llm,
        cache=cache,
        allow_new_topics=allow_new_topics,
        llm_concurrency=llm_concurrency,
        show_progress=show_progress,
    )

    for repost, decision in zip(reposts, decisions):
        if not decision.primary_topic:
            decision.primary_topic = UNCLASSIFIED
        topics = [decision.primary_topic] + [t for t in decision.secondary_topics if t and t != decision.primary_topic]
        primary_folder = safe_filename(decision.primary_topic)
        stem = _repost_stem(repost, decision, used_stems)
        note_target = f"Reposts/{primary_folder}/{stem}"
        _write_repost_note(vault, repost, decision, note_target)
        author = _author_note_name(repost)
        ref = NoteRef(
            repost_id=repost.repost_id,
            original_id=repost.original_id,
            target=note_target,
            filename=stem,
            display=decision.title,
            primary_topic=decision.primary_topic,
            topics=topics,
            author=author,
            reposted_at=repost.reposted_at,
            summary=decision.summary,
            decision_source=decision.source,
        )
        assignments.append((repost, decision, ref))
        by_primary_topic[decision.primary_topic].append((repost, decision))
        for topic in topics:
            by_topic[topic].append(ref)
        by_author[author].append(ref)
        for domain in repost.domains:
            by_domain[domain].append(ref)
        for hashtag in repost.hashtags:
            by_hashtag[hashtag].append(ref)

    cache.save(vault / "Data" / "llm_cache.json")
    if cache_path != vault / "Data" / "llm_cache.json":
        cache.save(cache_path)

    topic_summaries: dict[str, TopicSummary] = {}
    primary_topics_sorted = [topic for topic, _ in Counter(ref.primary_topic for _, _, ref in assignments).most_common()]
    # Include secondary-only topics too, sorted after primary topics.
    for topic, refs in sorted(by_topic.items(), key=lambda item: len(item[1]), reverse=True):
        if topic not in primary_topics_sorted:
            primary_topics_sorted.append(topic)
    topic_iterator = _progress(
        primary_topics_sorted,
        total=len(primary_topics_sorted),
        desc="LLM topic summaries" if llm else "Topic summaries",
        unit="topic",
        enabled=show_progress,
    )
    for topic in topic_iterator:
        if topic in by_primary_topic:
            topic_summaries[topic] = summarize_topic_with_llm(topic, by_primary_topic[topic], llm=llm, cache=cache)
        else:
            topic_summaries[topic] = TopicSummary(topic=topic, summary="This topic appears only as a secondary topic in this vault.", source="secondary_only")

    for topic, refs in by_topic.items():
        _write_topic_note(vault, topic, refs, topic_summaries[topic])
    for author, refs in by_author.items():
        _write_author_note(vault, author, refs)
    for domain, refs in by_domain.items():
        _write_simple_index_note(vault, "Domains", domain, "Domain", refs)
    for hashtag, refs in by_hashtag.items():
        _write_simple_index_note(vault, "Hashtags", hashtag, "Hashtag", refs)

    raw_lines = [json.dumps(repost.to_dict(), ensure_ascii=False) for repost, _, _ in assignments]
    write_text(vault / "Data" / "reposts.jsonl", "\n".join(raw_lines) + ("\n" if raw_lines else ""))

    assignment_lines = []
    for repost, decision, ref in assignments:
        assignment_lines.append(json.dumps({
            "repost_id": repost.repost_id,
            "original_id": repost.original_id,
            "note_target": ref.target,
            "title": decision.title,
            "summary": decision.summary,
            "primary_topic": decision.primary_topic,
            "secondary_topics": decision.secondary_topics,
            "confidence": decision.confidence,
            "source": decision.source,
            "author": ref.author,
            "reposted_at": repost.reposted_at,
            "original_url": repost.original_url,
        }, ensure_ascii=False))
    write_text(vault / "Data" / "topic_assignments.jsonl", "\n".join(assignment_lines) + ("\n" if assignment_lines else ""))
    write_json(vault / "Data" / "topic_summaries.json", {topic: summary.to_dict() for topic, summary in topic_summaries.items()})

    topics_metadata = []
    for i, topic in enumerate(primary_topics_sorted):
        topics_metadata.append({
            "topic": topic,
            "slug": slugify(topic),
            "cluster_tag": _topic_cluster_tag(topic),
            "topic_tag": _topic_note_tag(topic),
            "count_all": len(by_topic.get(topic, [])),
            "count_primary": len(by_primary_topic.get(topic, [])),
            "graph_color": PALETTE_HEX[i % len(PALETTE_HEX)],
        })
    write_json(vault / "Data" / "topics.json", topics_metadata)

    manifest = {
        "schema_version": "2.0",
        "reposts": len(reposts),
        "notes": len(assignments),
        "topics": len(by_topic),
        "primary_topics": len(by_primary_topic),
        "authors": len(by_author),
        "domains": len(by_domain),
        "hashtags": len(by_hashtag),
        "llm_model": llm.model if llm else None,
        "llm_base_url": llm.base_url if llm else None,
        "llm_enabled": llm is not None,
        "allow_new_topics": allow_new_topics,
        "llm_concurrency": max(1, int(llm_concurrency or 1)),
    }
    write_json(vault / "Data" / "manifest.json", manifest)

    cache.save(vault / "Data" / "llm_cache.json")
    if cache_path != vault / "Data" / "llm_cache.json":
        cache.save(cache_path)

    _write_graph_config(vault, primary_topics_sorted)
    _write_graph_guide(vault, primary_topics_sorted)

    summary_lines = [
        "# X Repost Interest Summary",
        "",
        f"Total reposts in this vault: {len(reposts)}",
        f"LLM enabled: {'yes' if llm else 'no'}",
        f"LLM model: {llm.model if llm else 'none'}",
        f"LLM concurrency: {max(1, int(llm_concurrency or 1)) if llm else 0}",
        "",
        "## Top primary topics",
        "",
    ]
    primary_counts = Counter(ref.primary_topic for _, _, ref in assignments)
    for topic, count in primary_counts.most_common(50):
        summary_lines.append(f"- [[Topics/{safe_filename(topic)}|{topic}]] — {count} primary reposts")
    summary_lines.extend(["", "## Top original authors", ""])
    for author, refs in sorted(by_author.items(), key=lambda item: len(item[1]), reverse=True)[:50]:
        summary_lines.append(f"- [[Authors/{safe_filename(author)}|{author}]] — {len(refs)}")
    summary_lines.extend(["", "## Top linked domains", ""])
    for domain, refs in sorted(by_domain.items(), key=lambda item: len(item[1]), reverse=True)[:50]:
        summary_lines.append(f"- [[Domains/{safe_filename(domain)}|{domain}]] — {len(refs)}")
    summary_lines.extend(["", "## Data files", "", "- [[Data/README|Data README]]", "- [[Graph Guide]]"])
    write_text(vault / "Interest Summary.md", "\n".join(summary_lines) + "\n")

    return {
        "reposts": len(reposts),
        "topics": len(by_topic),
        "primary_topics": len(by_primary_topic),
        "authors": len(by_author),
        "domains": len(by_domain),
        "hashtags": len(by_hashtag),
    }
