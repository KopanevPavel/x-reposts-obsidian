from xreposts_obsidian.classify import DEFAULT_TOPIC_RULES, infer_topics
from xreposts_obsidian.models import Repost, XUser


def _repost(text: str) -> Repost:
    return Repost(
        repost_id="1",
        reposted_at="2026-01-01T00:00:00Z",
        original_id="2",
        original_created_at="2026-01-01T00:00:00Z",
        original_text=text,
        original_author=XUser(username="robotlab", name="Robot Lab"),
        original_url="https://x.com/robotlab/status/2",
    )


def test_infer_robotics_topic():
    rules = {"Robotics": ["robotics", "manipulation", "planning"], "Cooking": ["recipe"]}
    assert infer_topics(_repost("A new robotics foundation model for manipulation and planning."), rules) == ["Robotics"]


def test_infer_gpu_topic():
    topics = infer_topics(_repost("CUDA kernels on Blackwell GPUs are speeding up inference."), DEFAULT_TOPIC_RULES)
    assert "GPU" in topics
