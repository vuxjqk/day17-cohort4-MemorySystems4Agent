from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore, extract_profile_updates

ROOT = Path(__file__).resolve().parent.parent


def make_config(tmp_path: Path, threshold: int = 300, keep: int = 4):
    """Isolated config: state goes to tmp_path, small compact threshold so compaction happens fast."""

    config = load_config(ROOT)
    return replace(
        config,
        state_dir=tmp_path / "state",
        compact_threshold_tokens=threshold,
        compact_keep_messages=keep,
        mode="offline",
    )


def _stress_turns() -> list[str]:
    data = json.loads((ROOT / "data" / "advanced_long_context.json").read_text(encoding="utf-8"))
    return data[0]["turns"]


# --- User.md -----------------------------------------------------------------


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    assert store.file_size("dungct") == 0
    assert "## Profile" in store.read_text("dungct")  # default template before any write

    path = store.write_text("dungct", "# User.md\n\n## Profile\n- name: DũngCT\n")
    assert path.exists() and path.name == "User.md"
    assert "DũngCT" in store.read_text("dungct")
    assert store.file_size("dungct") == path.stat().st_size > 0

    assert store.edit_text("dungct", "DũngCT", "DũngCT Stress") is True
    assert "DũngCT Stress" in store.read_text("dungct")
    assert store.edit_text("dungct", "không tồn tại", "x") is False


def test_user_id_is_sanitized(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    path = store.path_for("../../evil user")
    assert path.resolve().is_relative_to((tmp_path / "profiles").resolve())


def test_structured_facts_and_correction(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles")
    store.ingest_message("u", "Mình ở Đà Nẵng và đang làm backend engineer cho startup AI.")
    store.ingest_message("u", "Giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.")
    store.ingest_message("u", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")

    facts = store.facts("u")
    assert facts["location"] == "Huế"
    assert facts["profession"] == "MLOps engineer"
    text = store.read_text("u")
    # conflict handling: old value only survives in the Corrections log, not as a current fact
    assert "- location: Huế" in text and "- location: Đà Nẵng" not in text
    assert "`backend engineer` -> `MLOps engineer`" in text


# --- extraction guardrails (bonus) ------------------------------------------


def test_extraction_skips_questions_noise_and_negation() -> None:
    assert extract_profile_updates("Bạn có thể nhắc lại tên mình không?") == {}
    assert extract_profile_updates("Hiện tại mình làm nghề gì và mình còn ở Huế không?") == {}
    noisy = (
        "Có lúc mình đùa với đồng nghiệp rằng hay là chuyển sang product manager cho đỡ phải ngồi canh "
        "pipeline, nhưng đó chỉ là câu đùa. Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày."
    )
    assert "profession" not in extract_profile_updates(noisy)
    assert "location" not in extract_profile_updates(noisy)
    # the corgi's name must not be mistaken for the user's name
    assert extract_profile_updates("Mình nuôi một bé corgi tên Bơ.") == {"pet": "corgi tên Bơ"}


def test_confidence_threshold_blocks_hedged_or_stale_facts() -> None:
    # "Lúc đầu mình nói ở Huế" is stale -> low confidence; the later Đà Nẵng mention wins.
    msg = "Lúc đầu mình nói hiện ở Huế, nhưng thực ra từ tuần này mình đang làm việc ở Đà Nẵng vài tháng."
    assert extract_profile_updates(msg)["location"] == "Đà Nẵng"
    hedged = "Nếu bạn giải thích, hãy trả lời ngắn gọn và có ví dụ thực tế."
    assert "response_style" not in extract_profile_updates(hedged, threshold=0.6)
    assert "response_style" in extract_profile_updates(hedged, threshold=0.4)


def test_interest_decay_caps_memory_growth(tmp_path: Path) -> None:
    store = UserProfileStore(tmp_path / "profiles", max_interests=3, interest_decay=0.5)
    store.apply_updates("u", {"interests": "Python"})
    store.apply_updates("u", {"interests": "Python"})
    for topic in ["Rust", "Go", "Elixir", "Haskell"]:
        store.apply_updates("u", {"interests": topic})
    top = store.interests("u", limit=3)
    assert len(top) == 3
    assert "Haskell" in top  # most recent
    assert "Python" not in top  # mentioned twice, but long ago -> decayed out


# --- compact memory ------------------------------------------------------------


def test_compact_trigger(tmp_path: Path) -> None:
    manager = CompactMemoryManager(threshold_tokens=200, keep_messages=4)
    message = "nội dung khá dài " * 10
    for i in range(12):
        manager.append("t", "user", f"Lượt {i}: {message}")
    ctx = manager.context("t")
    assert manager.compaction_count("t") >= 1
    assert len(ctx["messages"]) <= 4 + 1
    assert "Lượt" in str(ctx["summary"])  # old content lives on in the summary
    bounded = manager.context_tokens("t")
    for i in range(12, 60):
        manager.append("t", "user", f"Lượt {i}: {message}")
    # context no longer grows with thread length (summary is capped, recent window is fixed)
    assert manager.context_tokens("t") <= bounded + 5

    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for turn in _stress_turns():
        agent.reply("dungct_stress", "stress", turn)
    assert agent.compaction_count("stress") >= 2


def test_no_compaction_on_short_thread(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path, threshold=1000), force_offline=True)
    for turn in ["Chào bạn, mình tên là DũngCT.", "Mình ở Đà Nẵng.", "Mình thích Python."]:
        agent.reply("dungct", "short", turn)
    assert agent.compaction_count("short") == 0


# --- cross-session recall ------------------------------------------------------


def test_cross_session_recall(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    turns = [
        "Chào bạn, mình tên là DũngCT.",
        "Đồ uống yêu thích là cà phê sữa đá.",
        "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.",
    ]
    for turn in turns:
        baseline.reply("dungct", "s1", turn)
        advanced.reply("dungct", "s1", turn)

    question = "Mình tên gì, làm nghề gì và đồ uống yêu thích là gì?"

    # Baseline remembers inside the same thread ...
    same_thread = baseline.reply("dungct", "s1", question)["response"]
    assert "DũngCT" in same_thread
    # ... but forgets in a new thread.
    new_thread = baseline.reply("dungct", "s2", question)["response"]
    assert "DũngCT" not in new_thread and "cà phê sữa đá" not in new_thread
    assert baseline.memory_file_size("dungct") == 0

    # Advanced recalls from User.md, even from a brand-new agent instance (new process / session).
    fresh_advanced = AdvancedAgent(config, force_offline=True)
    answer = fresh_advanced.reply("dungct", "s2", question)["response"]
    assert "DũngCT" in answer and "cà phê sữa đá" in answer and "MLOps engineer" in answer
    assert "backend engineer" not in answer


def test_stress_recall_prefers_latest_facts(tmp_path: Path) -> None:
    agent = AdvancedAgent(make_config(tmp_path), force_offline=True)
    for turn in _stress_turns():
        agent.reply("dungct_stress", "stress", turn)
    answer = agent.reply(
        "dungct_stress", "new", "Nếu ai đó nhắc Huế, Hà Nội hay product manager, đâu mới là nghề nghiệp và nơi ở hiện tại của mình?"
    )["response"]
    assert "MLOps engineer" in answer and "Đà Nẵng" in answer
    assert "Huế" not in answer and "Hà Nội" not in answer and "product manager" not in answer


# --- prompt load -----------------------------------------------------------------


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    config = make_config(tmp_path, threshold=1000)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for turn in _stress_turns():
        baseline.reply("dungct_stress", "long", turn)
        advanced.reply("dungct_stress", "long", turn)

    assert advanced.compaction_count("long") >= 1
    assert advanced.prompt_token_usage("long") < baseline.prompt_token_usage("long") * 0.75


def test_advanced_costs_more_prompt_on_short_thread(tmp_path: Path) -> None:
    """Trade-off: without compaction, User.md is pure overhead on short threads."""

    config = make_config(tmp_path, threshold=1000)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)
    for turn in ["Chào bạn, mình tên là DũngCT.", "Mình ở Đà Nẵng và đang làm backend engineer.", "Mình thích Python."]:
        baseline.reply("dungct", "short", turn)
        advanced.reply("dungct", "short", turn)
    assert advanced.prompt_token_usage("short") > baseline.prompt_token_usage("short")
