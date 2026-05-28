from __future__ import annotations

import random
import time
from typing import Any, Iterator

import requests

from .auth import OAuth2PKCEAuthenticator

API_BASE = "https://api.x.com/2"


class XApiError(RuntimeError):
    pass


class XClient:
    def __init__(
        self,
        authenticator: OAuth2PKCEAuthenticator,
        sleep_on_rate_limit: bool = True,
        request_timeout_seconds: float | None = None,
        max_retries: int = 6,
        retry_backoff_seconds: float = 2.0,
    ):
        self.authenticator = authenticator
        self.sleep_on_rate_limit = sleep_on_rate_limit
        self.request_timeout_seconds = (
            float(request_timeout_seconds)
            if request_timeout_seconds is not None
            else float(self.authenticator.settings.request_timeout_seconds)
        )
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.1, float(retry_backoff_seconds))

    def _sleep_before_retry(self, attempt: int, reason: str) -> None:
        # Exponential backoff with small jitter. attempt starts at 1.
        delay = min(120.0, self.retry_backoff_seconds * (2 ** (attempt - 1)))
        delay += random.uniform(0.0, min(1.0, delay * 0.1))
        print(f"X API request failed ({reason}). Retry {attempt}/{self.max_retries} in {delay:.1f}s.")
        time.sleep(delay)

    def request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        retry_auth: bool = True,
    ) -> dict[str, Any]:
        url = API_BASE + path
        retryable_statuses = {500, 502, 503, 504}

        for attempt_index in range(self.max_retries + 1):
            headers = {
                "Authorization": f"Bearer {self.authenticator.get_access_token()}",
                "User-Agent": "x-reposts-obsidian/0.5.0",
            }

            try:
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    timeout=self.request_timeout_seconds,
                )
            except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
                if attempt_index >= self.max_retries:
                    raise XApiError(
                        "X API network request failed after "
                        f"{self.max_retries + 1} attempts: {type(exc).__name__}: {exc}"
                    ) from exc
                self._sleep_before_retry(attempt_index + 1, f"{type(exc).__name__}")
                continue

            if response.status_code == 401 and retry_auth:
                token = self.authenticator.token_store.load()
                refresh_token = token.get("refresh_token")
                if refresh_token:
                    self.authenticator.refresh_access_token(str(refresh_token))
                    return self.request(method, path, params=params, retry_auth=False)

            if response.status_code == 429 and self.sleep_on_rate_limit:
                reset = int(response.headers.get("x-rate-limit-reset") or "0")
                wait_seconds = max(1, reset - int(time.time()) + 2)
                print(f"Rate limit hit. Sleeping {wait_seconds} seconds until X resets this window.")
                time.sleep(wait_seconds)
                return self.request(method, path, params=params, retry_auth=retry_auth)

            if response.status_code in retryable_statuses:
                if attempt_index >= self.max_retries:
                    raise XApiError(f"X API HTTP {response.status_code} after retries: {response.text}")
                self._sleep_before_retry(attempt_index + 1, f"HTTP {response.status_code}")
                continue

            if response.status_code >= 400:
                raise XApiError(f"X API HTTP {response.status_code}: {response.text}")

            if not response.text.strip():
                return {}
            return response.json()

        raise XApiError("X API request failed unexpectedly after retry loop.")

    def get_me(self) -> dict[str, Any]:
        params = {"user.fields": "created_at,description,id,name,protected,username,verified"}
        return self.request("GET", "/users/me", params=params)

    def iter_user_posts(
        self,
        user_id: str,
        max_posts: int = 3200,
        start_time: str | None = None,
        end_time: str | None = None,
        since_id: str | None = None,
        until_id: str | None = None,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        params: dict[str, Any] = {
            "max_results": 100,
            "tweet.fields": ",".join(
                [
                    "author_id",
                    "context_annotations",
                    "conversation_id",
                    "created_at",
                    "entities",
                    "id",
                    "lang",
                    "note_tweet",
                    "public_metrics",
                    "referenced_tweets",
                    "text",
                ]
            ),
            "expansions": ",".join(
                [
                    "author_id",
                    "entities.mentions.username",
                    "referenced_tweets.id",
                    "referenced_tweets.id.author_id",
                ]
            ),
            "user.fields": "created_at,description,id,name,protected,public_metrics,username,verified,verified_type,url",
        }
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        if since_id:
            params["since_id"] = since_id
        if until_id:
            params["until_id"] = until_id

        fetched = 0
        page_count = 0
        next_token: str | None = None

        while True:
            if next_token:
                params["pagination_token"] = next_token
            elif "pagination_token" in params:
                del params["pagination_token"]

            page = self.request("GET", f"/users/{user_id}/tweets", params=params)
            page_count += 1
            data = page.get("data") or []
            fetched += len(data)
            yield page

            meta = page.get("meta") or {}
            next_token = meta.get("next_token")
            if not next_token:
                break
            if fetched >= max_posts:
                break
            if max_pages is not None and page_count >= max_pages:
                break
