from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .auth import OAuth2PKCEAuthenticator
from .classify import load_topic_rules
from .client import XClient
from .llm import DEFAULT_LLM_MODEL, DEFAULT_OLLAMA_BASE_URL, OllamaClient
from .models import extract_reposts_from_page
from .obsidian import build_vault
from .settings import load_settings, require_client_id
from .utils import write_text


def _add_common_env_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", default=None, help="Path to .env file. Defaults to .env in the current directory.")


def _add_llm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm-mode",
        choices=["auto", "required", "off"],
        default=os.getenv("XRO_LLM_MODE", "auto"),
        help="auto: use Ollama if reachable; required: fail if unavailable; off: use keyword rules only.",
    )
    parser.add_argument("--llm-model", default=os.getenv("XRO_LLM_MODEL", DEFAULT_LLM_MODEL), help="Ollama model name.")
    parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL), help="Ollama HTTP base URL, e.g. http://DGX_SPARK_IP:11434.")
    parser.add_argument("--llm-timeout", type=float, default=float(os.getenv("XRO_LLM_TIMEOUT", "240")), help="Seconds to wait for one LLM call.")
    parser.add_argument("--llm-cache", default=os.getenv("XRO_LLM_CACHE", None), help="Optional external JSON cache path. The vault also receives Data/llm_cache.json.")
    parser.add_argument("--llm-concurrency", type=int, default=int(os.getenv("XRO_LLM_CONCURRENCY", "1")), help="Number of concurrent Ollama classification requests during build. Use 1 for sequential mode.")
    parser.add_argument("--no-new-topics", action="store_true", help="Disallow LLM-created topics; force choices from config/topics.yaml plus Unclassified.")




def _add_x_api_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-timeout", type=float, default=float(os.getenv("XRO_REQUEST_TIMEOUT_SECONDS", "30.0")), help="Seconds to wait for one X API request before retrying.")
    parser.add_argument("--api-max-retries", type=int, default=int(os.getenv("XRO_API_MAX_RETRIES", "6")), help="Retry count for X API network timeouts and HTTP 5xx responses.")
    parser.add_argument("--api-retry-backoff", type=float, default=float(os.getenv("XRO_API_RETRY_BACKOFF", "2.0")), help="Initial exponential backoff delay in seconds for retryable X API failures.")

def _make_llm(args: argparse.Namespace) -> OllamaClient | None:
    if getattr(args, "llm_mode", "auto") == "off":
        print("LLM mode is off. Using keyword rules only.")
        return None
    llm = OllamaClient(base_url=args.ollama_base_url, model=args.llm_model, timeout_seconds=args.llm_timeout)
    try:
        info = llm.check()
        available = info.get("model_available", False)
        print(f"Ollama reachable at {llm.base_url}. Model requested: {llm.model}.")
        if not available:
            print("Warning: requested model was not listed by Ollama. If the next call fails, run: ollama pull " + llm.model)
        return llm
    except Exception as exc:  # noqa: BLE001
        message = f"Could not reach Ollama at {llm.base_url}: {type(exc).__name__}: {exc}"
        if args.llm_mode == "required":
            raise SystemExit(message)
        print("Warning: " + message)
        print("Falling back to keyword rules. Use --llm-mode required to fail instead.")
        return None


def cmd_auth(args: argparse.Namespace) -> None:
    settings = load_settings(args.env)
    require_client_id(settings)
    auth = OAuth2PKCEAuthenticator(settings)
    auth.login(open_browser=not args.no_browser, timeout_seconds=args.timeout)


def cmd_status(args: argparse.Namespace) -> None:
    settings = load_settings(args.env)
    require_client_id(settings)
    auth = OAuth2PKCEAuthenticator(settings)
    client = XClient(auth, sleep_on_rate_limit=not args.no_rate_limit_sleep, request_timeout_seconds=getattr(args, "api_timeout", None), max_retries=getattr(args, "api_max_retries", 6), retry_backoff_seconds=getattr(args, "api_retry_backoff", 2.0))
    me = client.get_me().get("data") or {}
    if not me:
        raise SystemExit("Authenticated request succeeded, but /2/users/me did not return user data.")
    print(json.dumps(me, ensure_ascii=False, indent=2))


def cmd_llm_test(args: argparse.Namespace) -> None:
    llm = OllamaClient(base_url=args.ollama_base_url, model=args.llm_model, timeout_seconds=args.llm_timeout)
    info = llm.check()
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if not info.get("model_available"):
        print("\nModel is not listed locally. Pull it on the machine running Ollama:")
        print(f"ollama pull {args.llm_model}")
        return
    response = llm.chat_json(
        [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": "Return {\"ok\": true, \"topic\": \"GPU\", \"title\": \"Local LLM Test\"}."},
        ],
        temperature=0.0,
    )
    print("\nLLM JSON response:")
    print(json.dumps(response, ensure_ascii=False, indent=2))


def cmd_build(args: argparse.Namespace) -> None:
    settings = load_settings(args.env)
    require_client_id(settings)
    auth = OAuth2PKCEAuthenticator(settings)
    client = XClient(auth, sleep_on_rate_limit=not args.no_rate_limit_sleep, request_timeout_seconds=getattr(args, "api_timeout", None), max_retries=getattr(args, "api_max_retries", 6), retry_backoff_seconds=getattr(args, "api_retry_backoff", 2.0))
    rules = load_topic_rules(args.topics)
    llm = _make_llm(args)

    if args.user_id:
        user_id = args.user_id
        username = ""
    else:
        me = client.get_me().get("data") or {}
        user_id = str(me.get("id") or "")
        username = str(me.get("username") or "")
        if not user_id:
            raise SystemExit("Could not resolve authenticated user ID from /2/users/me.")

    print(f"Fetching posts for user_id={user_id}" + (f" (@{username})" if username else ""))
    print("Retweets/reposts are detected from referenced_tweets.type == 'retweeted'.")

    all_reposts = []
    pages = 0
    fetched_posts = 0

    for page in client.iter_user_posts(
        user_id=user_id,
        max_posts=args.max_posts,
        start_time=args.start_time,
        end_time=args.end_time,
        since_id=args.since_id,
        until_id=args.until_id,
        max_pages=args.max_pages,
    ):
        pages += 1
        fetched_posts += len(page.get("data") or [])
        page_reposts = extract_reposts_from_page(page)
        all_reposts.extend(page_reposts)
        print(f"Page {pages}: posts={len(page.get('data') or [])}, reposts_on_page={len(page_reposts)}, total_reposts={len(all_reposts)}")

        if fetched_posts >= args.max_posts:
            break

    stats = build_vault(
        reposts=all_reposts,
        vault_path=args.vault,
        topic_rules=rules,
        min_topic_score=args.min_topic_score,
        clear=args.clear,
        llm=llm,
        llm_cache_path=args.llm_cache,
        allow_new_topics=not args.no_new_topics,
        llm_concurrency=args.llm_concurrency,
        show_progress=not args.no_progress,
    )

    coverage = {
        "user_id": user_id,
        "username": username,
        "pages_fetched": pages,
        "posts_fetched": fetched_posts,
        "reposts_found": len(all_reposts),
        "max_posts_requested": args.max_posts,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "since_id": args.since_id,
        "until_id": args.until_id,
        "llm_mode": args.llm_mode,
        "llm_used": llm is not None,
        "llm_model": llm.model if llm else None,
        "ollama_base_url": llm.base_url if llm else None,
        "llm_concurrency": max(1, int(args.llm_concurrency or 1)) if llm else 0,
        "api_timeout_seconds": args.api_timeout,
        "api_max_retries": args.api_max_retries,
        "api_retry_backoff_seconds": args.api_retry_backoff,
        "note": "X user-posts timeline retrieval is limited by the X API plan and the timeline endpoint's retrievable history. The documented user-posts timeline cap is 3,200 most recent posts.",
    }
    write_text(Path(args.vault).expanduser().resolve() / "Data" / "coverage.json", json.dumps(coverage, ensure_ascii=False, indent=2) + "\n")

    print("\nDone.")
    print(f"Vault: {Path(args.vault).expanduser().resolve()}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("Open that folder as an Obsidian vault and start from 'Interest Summary.md' and 'Graph Guide.md'.")


def cmd_export_json(args: argparse.Namespace) -> None:
    settings = load_settings(args.env)
    require_client_id(settings)
    auth = OAuth2PKCEAuthenticator(settings)
    client = XClient(auth, sleep_on_rate_limit=not args.no_rate_limit_sleep, request_timeout_seconds=getattr(args, "api_timeout", None), max_retries=getattr(args, "api_max_retries", 6), retry_backoff_seconds=getattr(args, "api_retry_backoff", 2.0))

    if args.user_id:
        user_id = args.user_id
    else:
        me = client.get_me().get("data") or {}
        user_id = str(me.get("id") or "")
        if not user_id:
            raise SystemExit("Could not resolve authenticated user ID from /2/users/me.")

    all_reposts = []
    fetched_posts = 0
    for page in client.iter_user_posts(
        user_id=user_id,
        max_posts=args.max_posts,
        start_time=args.start_time,
        end_time=args.end_time,
        since_id=args.since_id,
        until_id=args.until_id,
        max_pages=args.max_pages,
    ):
        fetched_posts += len(page.get("data") or [])
        all_reposts.extend(extract_reposts_from_page(page))
        if fetched_posts >= args.max_posts:
            break

    out = Path(args.out).expanduser().resolve()
    lines = [json.dumps(repost.to_dict(), ensure_ascii=False) for repost in all_reposts]
    write_text(out, "\n".join(lines) + ("\n" if lines else ""))
    print(f"Wrote {len(all_reposts)} reposts to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xro",
        description="Fetch your X reposts via X API v2 and generate an Obsidian graph vault.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    auth_parser = sub.add_parser("auth", help="Run OAuth 2.0 PKCE login and save a local token.")
    _add_common_env_arg(auth_parser)
    auth_parser.add_argument("--no-browser", action="store_true", help="Print the auth URL without opening the browser.")
    auth_parser.add_argument("--timeout", type=int, default=180, help="Seconds to wait for the OAuth callback.")
    auth_parser.set_defaults(func=cmd_auth)

    status_parser = sub.add_parser("status", help="Print the authenticated X user returned by /2/users/me.")
    _add_common_env_arg(status_parser)
    _add_x_api_args(status_parser)
    status_parser.add_argument("--no-rate-limit-sleep", action="store_true", help="Do not sleep and retry on HTTP 429.")
    status_parser.set_defaults(func=cmd_status)

    llm_parser = sub.add_parser("llm-test", help="Check Ollama connectivity and test JSON generation.")
    llm_parser.add_argument("--llm-model", default=os.getenv("XRO_LLM_MODEL", DEFAULT_LLM_MODEL))
    llm_parser.add_argument("--ollama-base-url", default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL))
    llm_parser.add_argument("--llm-timeout", type=float, default=float(os.getenv("XRO_LLM_TIMEOUT", "240")))
    llm_parser.set_defaults(func=cmd_llm_test)

    build = sub.add_parser("build", help="Fetch reposts and create an Obsidian vault.")
    _add_common_env_arg(build)
    _add_x_api_args(build)
    _add_llm_args(build)
    build.add_argument("--vault", default="X-Reposts-Obsidian", help="Output folder to use as the Obsidian vault.")
    build.add_argument("--topics", default="config/topics.yaml", help="YAML topic rules file. These are candidate topics; the LLM may create new topics unless --no-new-topics is set.")
    build.add_argument("--min-topic-score", type=int, default=1, help="Minimum keyword score required for fallback classification.")
    build.add_argument("--max-posts", type=int, default=3200, help="Maximum user timeline posts to scan. X documents a 3,200-post user timeline cap.")
    build.add_argument("--max-pages", type=int, default=None, help="Optional page cap for testing. Each page has up to 100 posts.")
    build.add_argument("--start-time", default=None, help="Earliest post timestamp, UTC ISO 8601, e.g. 2025-01-01T00:00:00Z.")
    build.add_argument("--end-time", default=None, help="Latest post timestamp, UTC ISO 8601, e.g. 2026-01-01T00:00:00Z.")
    build.add_argument("--since-id", default=None, help="Return posts after this post ID. Takes precedence over start_time.")
    build.add_argument("--until-id", default=None, help="Return posts before this post ID. Takes precedence over end_time.")
    build.add_argument("--user-id", default=None, help="Override authenticated user ID. Usually not needed.")
    build.add_argument("--clear", action="store_true", help="Delete the vault folder before writing new notes.")
    build.add_argument("--no-rate-limit-sleep", action="store_true", help="Do not sleep and retry on HTTP 429.")
    build.add_argument("--no-progress", action="store_true", help="Disable progress bars during LLM classification and topic summarization.")
    build.set_defaults(func=cmd_build)

    export_json = sub.add_parser("export-json", help="Fetch reposts and write JSON Lines without creating Markdown notes.")
    _add_common_env_arg(export_json)
    _add_x_api_args(export_json)
    export_json.add_argument("--out", default="reposts.jsonl", help="Output JSONL file.")
    export_json.add_argument("--max-posts", type=int, default=3200)
    export_json.add_argument("--max-pages", type=int, default=None)
    export_json.add_argument("--start-time", default=None)
    export_json.add_argument("--end-time", default=None)
    export_json.add_argument("--since-id", default=None)
    export_json.add_argument("--until-id", default=None)
    export_json.add_argument("--user-id", default=None)
    export_json.add_argument("--no-rate-limit-sleep", action="store_true")
    export_json.set_defaults(func=cmd_export_json)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
