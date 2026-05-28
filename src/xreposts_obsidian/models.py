from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .utils import domain_from_url


@dataclass
class XUser:
    id: str = ""
    username: str = ""
    name: str = ""
    description: str = ""
    verified: bool = False


@dataclass
class Repost:
    repost_id: str
    reposted_at: str
    original_id: str
    original_created_at: str
    original_text: str
    original_author: XUser
    original_url: str
    hashtags: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    context_labels: list[str] = field(default_factory=list)
    public_metrics: dict[str, Any] = field(default_factory=dict)
    raw_repost: dict[str, Any] = field(default_factory=dict)
    raw_original: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tweet_text(tweet: dict[str, Any]) -> str:
    note_tweet = tweet.get("note_tweet")
    if isinstance(note_tweet, dict) and isinstance(note_tweet.get("text"), str):
        return note_tweet["text"]
    return str(tweet.get("text") or "")


def _entities(tweet: dict[str, Any]) -> dict[str, Any]:
    entities = tweet.get("entities")
    return entities if isinstance(entities, dict) else {}


def extract_urls(tweet: dict[str, Any]) -> list[str]:
    urls: set[str] = set()
    entities = _entities(tweet)
    entity_urls = entities.get("urls", []) or []
    for item in entity_urls:
        if not isinstance(item, dict):
            continue
        url = item.get("expanded_url") or item.get("unwound_url") or item.get("url")
        if url:
            urls.add(str(url))

    # X text commonly contains shortened t.co URLs. When entities are present,
    # expanded_url/unwound_url is more useful for Obsidian domain graphs, so only
    # fall back to regex extraction when the API gave no URL entities.
    if not urls:
        for url in re.findall(r"https?://[^\s)>\]]+", tweet_text(tweet)):
            urls.add(url)
    return sorted(urls)


def extract_hashtags(tweet: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    entities = _entities(tweet)
    for item in entities.get("hashtags", []) or []:
        if not isinstance(item, dict):
            continue
        tag = item.get("tag") or item.get("text")
        if tag:
            tags.add(str(tag).lower())
    for tag in re.findall(r"#([A-Za-z0-9_]+)", tweet_text(tweet)):
        tags.add(tag.lower())
    return sorted(tags)


def extract_mentions(tweet: dict[str, Any]) -> list[str]:
    mentions: set[str] = set()
    entities = _entities(tweet)
    for item in entities.get("mentions", []) or []:
        if not isinstance(item, dict):
            continue
        username = item.get("username") or item.get("screen_name")
        if username:
            mentions.add("@" + str(username).lower().lstrip("@"))
    for username in re.findall(r"@([A-Za-z0-9_]{1,20})", tweet_text(tweet)):
        mentions.add("@" + username.lower())
    return sorted(mentions)


def extract_context_labels(tweet: dict[str, Any]) -> list[str]:
    labels: set[str] = set()
    for annotation in tweet.get("context_annotations", []) or []:
        if not isinstance(annotation, dict):
            continue
        domain = annotation.get("domain")
        entity = annotation.get("entity")
        if isinstance(domain, dict) and domain.get("name"):
            labels.add(str(domain["name"]))
        if isinstance(entity, dict) and entity.get("name"):
            labels.add(str(entity["name"]))
    return sorted(labels)


def user_from_dict(data: dict[str, Any] | None) -> XUser:
    data = data or {}
    return XUser(
        id=str(data.get("id") or ""),
        username=str(data.get("username") or ""),
        name=str(data.get("name") or ""),
        description=str(data.get("description") or ""),
        verified=bool(data.get("verified") or False),
    )


def fallback_author_from_text(text: str) -> XUser:
    match = re.match(r"RT @([A-Za-z0-9_]{1,20}):", text or "")
    if not match:
        return XUser()
    username = match.group(1)
    return XUser(username=username, name=username)


def post_url(username: str, post_id: str) -> str:
    if username:
        return f"https://x.com/{username}/status/{post_id}"
    return f"https://x.com/i/web/status/{post_id}"


def extract_reposts_from_page(page: dict[str, Any]) -> list[Repost]:
    includes = page.get("includes") or {}
    included_tweets = {str(tweet.get("id")): tweet for tweet in includes.get("tweets", []) or [] if isinstance(tweet, dict)}
    included_users = {str(user.get("id")): user for user in includes.get("users", []) or [] if isinstance(user, dict)}

    reposts: list[Repost] = []
    for post in page.get("data", []) or []:
        if not isinstance(post, dict):
            continue
        refs = post.get("referenced_tweets") or []
        retweeted_ref = None
        for ref in refs:
            if isinstance(ref, dict) and ref.get("type") in {"retweeted", "reposted"}:
                retweeted_ref = ref
                break
        if not retweeted_ref:
            continue

        original_id = str(retweeted_ref.get("id") or "")
        original = included_tweets.get(original_id, {"id": original_id, "text": tweet_text(post)})
        original_author = user_from_dict(included_users.get(str(original.get("author_id") or "")))
        if not original_author.username:
            original_author = fallback_author_from_text(tweet_text(post))

        combined_urls = sorted(set(extract_urls(original) + extract_urls(post)))
        combined_domains = sorted({domain_from_url(url) for url in combined_urls if domain_from_url(url)})
        combined_hashtags = sorted(set(extract_hashtags(original) + extract_hashtags(post)))
        combined_mentions = sorted(set(extract_mentions(original) + extract_mentions(post)))
        combined_context = sorted(set(extract_context_labels(original) + extract_context_labels(post)))

        reposts.append(
            Repost(
                repost_id=str(post.get("id") or ""),
                reposted_at=str(post.get("created_at") or ""),
                original_id=original_id,
                original_created_at=str(original.get("created_at") or ""),
                original_text=tweet_text(original),
                original_author=original_author,
                original_url=post_url(original_author.username, original_id),
                hashtags=combined_hashtags,
                mentions=combined_mentions,
                urls=combined_urls,
                domains=combined_domains,
                context_labels=combined_context,
                public_metrics=original.get("public_metrics") or {},
                raw_repost=post,
                raw_original=original,
            )
        )
    return reposts
