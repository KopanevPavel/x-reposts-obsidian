import threading
import time

from xreposts_obsidian import obsidian
from xreposts_obsidian.llm import TopicDecision
from xreposts_obsidian.models import Repost, XUser


class DummyLLM:
    model = "dummy"
    base_url = "http://dummy"


def _repost(i: int) -> Repost:
    return Repost(
        repost_id=str(i),
        reposted_at="2026-01-01T00:00:00Z",
        original_id=str(100 + i),
        original_created_at="2026-01-01T00:00:00Z",
        original_text=f"Tweet {i}",
        original_author=XUser(username="author", name="Author"),
        original_url=f"https://x.com/author/status/{100 + i}",
    )


def test_classify_reposts_concurrent_preserves_order(monkeypatch):
    seen_threads = set()

    def fake_classify_repost(repost, **kwargs):
        seen_threads.add(threading.current_thread().name)
        time.sleep(0.02)
        return TopicDecision(
            title=f"Title {repost.repost_id}",
            summary="summary",
            primary_topic="GPU",
            source="fake",
        )

    monkeypatch.setattr(obsidian, "classify_repost", fake_classify_repost)

    reposts = [_repost(i) for i in range(8)]
    decisions = obsidian._classify_reposts(
        reposts=reposts,
        topic_rules={"GPU": ["gpu"]},
        min_topic_score=1,
        llm=DummyLLM(),
        cache=None,
        allow_new_topics=True,
        llm_concurrency=4,
    )

    assert [d.title for d in decisions] == [f"Title {i}" for i in range(8)]
    assert len(seen_threads) > 1
