from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config

COLUMNS = [
    "Agent",
    "Agent tokens only",
    "Prompt tokens processed",
    "Cross-session recall",
    "Response quality",
    "Memory growth (bytes)",
    "Compactions",
]


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int
    details: list[dict[str, Any]] = field(default_factory=list, repr=False)


def load_conversations(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else [data]


def _contains(answer: str, needle: str) -> bool:
    return needle.casefold() in answer.casefold()


def recall_points(answer: str, expected: list[str]) -> float:
    """1 if every expected fact appears, 0.5 if some do, 0 if none."""

    if not expected:
        return 1.0
    hits = sum(_contains(answer, item) for item in expected)
    if hits == len(expected):
        return 1.0
    return 0.5 if hits else 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Offline quality score in [0, 1].

    - 70%: fraction of expected facts covered (precise, not all-or-nothing)
    - 20%: concise (<= 60 words full credit, decays after)
    - 10%: not a refusal / "chưa có thông tin"
    """

    coverage = sum(_contains(answer, item) for item in expected) / len(expected) if expected else 1.0
    words = len(answer.split())
    concise = 1.0 if words <= 60 else max(0.0, 1 - (words - 60) / 120)
    refusal = bool(re.search(r"chưa có (thông tin|fact)|không biết|không nhớ", answer, re.IGNORECASE))
    return round(0.7 * coverage + 0.2 * concise + 0.1 * (0.0 if refusal else 1.0), 3)


JUDGE_PROMPT = """Bạn là giám khảo chấm chất lượng câu trả lời của một AI agent có bộ nhớ.
Câu hỏi của người dùng (hỏi ở một thread mới): {question}
Các fact đúng mà câu trả lời cần nêu: {expected}
Câu trả lời của agent: {answer}

Chấm theo thang 0-10: đúng và đủ fact (quan trọng nhất), không nêu fact cũ/sai, ngắn gọn, tự nhiên.
Chỉ trả về một số nguyên từ 0 đến 10."""


def make_llm_judge(config):
    """Return judge(question, answer, expected) -> score in [0, 1] using `config.judge_model`."""

    from model_provider import build_chat_model, message_text

    model = build_chat_model(config.judge_model)

    def judge(question: str, answer: str, expected: list[str]) -> float:
        reply = message_text(
            model.invoke(JUDGE_PROMPT.format(question=question, expected=", ".join(expected), answer=answer))
        )
        match = re.search(r"\d+", reply)
        return round(min(10, int(match.group())) / 10, 3) if match else 0.0

    return judge


def run_agent_benchmark(
    agent_name: str, agent, conversations: list[dict[str, Any]], config, judge=None
) -> BenchmarkRow:
    """Feed every conversation to `agent`, then ask its recall questions in fresh threads."""

    user_ids = sorted({conv["user_id"] for conv in conversations})
    size_before = {uid: agent.memory_file_size(uid) if hasattr(agent, "memory_file_size") else 0 for uid in user_ids}

    agent_tokens = prompt_tokens = compactions = 0
    recalls: list[float] = []
    qualities: list[float] = []
    details: list[dict[str, Any]] = []

    for conv in conversations:
        user_id, thread_id = conv["user_id"], f"{conv['id']}-main"
        for turn in conv["turns"]:
            agent.reply(user_id, thread_id, turn)
        agent_tokens += agent.token_usage(thread_id)
        prompt_tokens += agent.prompt_token_usage(thread_id)
        compactions += agent.compaction_count(thread_id)

        for idx, item in enumerate(conv.get("recall_questions", []), start=1):
            recall_thread = f"{conv['id']}-recall-{idx}"  # new thread = new session
            answer = agent.reply(user_id, recall_thread, item["question"])["response"]
            agent_tokens += agent.token_usage(recall_thread)
            prompt_tokens += agent.prompt_token_usage(recall_thread)
            score = recall_points(answer, item["expected_contains"])
            if judge is not None:
                quality = judge(item["question"], answer, item["expected_contains"])
            else:
                quality = heuristic_quality(answer, item["expected_contains"])
            recalls.append(score)
            qualities.append(quality)
            details.append(
                {"conversation": conv["id"], "question": item["question"], "answer": answer, "recall": score, "quality": quality}
            )

    growth = sum(
        (agent.memory_file_size(uid) if hasattr(agent, "memory_file_size") else 0) - size_before[uid] for uid in user_ids
    )
    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(sum(recalls) / len(recalls), 3) if recalls else 0.0,
        response_quality=round(sum(qualities) / len(qualities), 3) if qualities else 0.0,
        memory_growth_bytes=growth,
        compactions=compactions,
        details=details,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    table = [
        [
            r.agent_name,
            r.agent_tokens_only,
            r.prompt_tokens_processed,
            f"{r.recall_score:.2f}",
            f"{r.response_quality:.2f}",
            r.memory_growth_bytes,
            r.compactions,
        ]
        for r in rows
    ]
    try:
        from tabulate import tabulate

        return tabulate(table, headers=COLUMNS, tablefmt="github")
    except ImportError:
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "|".join("---" for _ in COLUMNS) + "|"]
        lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in table]
        return "\n".join(lines)


def _ratio_note(rows: list[BenchmarkRow]) -> str:
    base, adv = rows
    if not base.prompt_tokens_processed:
        return ""
    delta = (adv.prompt_tokens_processed - base.prompt_tokens_processed) / base.prompt_tokens_processed * 100
    direction = "nhiều hơn" if delta > 0 else "ít hơn"
    return (
        f"Advanced xử lý {abs(delta):.1f}% prompt tokens {direction} Baseline; "
        f"recall {base.recall_score:.2f} -> {adv.recall_score:.2f}."
    )


def run_suite(
    title: str, dataset: Path, config, verbose: bool = False, live: bool = False, judge=None
) -> list[BenchmarkRow]:
    conversations = load_conversations(dataset)
    suite_state = config.state_dir / ("benchmark_live" if live else "benchmark") / dataset.stem
    shutil.rmtree(suite_state, ignore_errors=True)
    suite_config = replace(config, state_dir=suite_state)

    rows = []
    for name, cls in (("Baseline", BaselineAgent), ("Advanced", AdvancedAgent)):
        agent = cls(suite_config, force_offline=not live)
        if live and agent.langchain_agent is None:
            raise SystemExit(f"{name}: live agent unavailable (install langchain/langgraph + provider SDK).")
        rows.append(run_agent_benchmark(name, agent, conversations, suite_config, judge=judge))
    print(f"\n## {title}\n")
    print(f"Dataset: data/{dataset.name} ({len(conversations)} conversation(s), "
          f"{sum(len(c['turns']) for c in conversations)} turns)\n")
    print(format_rows(rows))
    print("\n" + _ratio_note(rows))
    if verbose:
        for row in rows:
            print(f"\n### {row.agent_name} answers")
            for d in row.details:
                answer = d["answer"].replace("\n", " / ")
                print(f"- [{d['conversation']}] recall={d['recall']} | Q: {d['question']}\n    A: {answer}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline vs Advanced memory benchmark")
    parser.add_argument("--verbose", "-v", action="store_true", help="print every recall answer")
    parser.add_argument("--live", action="store_true", help="use the real LLM (LangChain) instead of offline mode")
    parser.add_argument("--no-judge", action="store_true", help="live mode: use heuristic quality instead of LLM judge")
    parser.add_argument("--only", choices=["standard", "stress"], help="run a single suite")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    config = load_config(Path(__file__).resolve().parent.parent)
    judge = None
    if args.live:
        config = replace(config, mode="live")
        if not config.model.is_configured():
            raise SystemExit(f"Provider {config.model.provider} is not configured (missing API key / base URL).")
        if not args.no_judge:
            judge = make_llm_judge(config)
    mode = f"live, {config.model.provider}/{config.model.model_name}" if args.live else "offline"
    quality = f"LLM judge {config.judge_model.model_name}" if judge else "heuristic"
    print(f"# Memory benchmark (mode={mode}, quality={quality}, compact threshold={config.compact_threshold_tokens} "
          f"tokens, keep={config.compact_keep_messages} messages)")
    if args.only != "stress":
        run_suite("Standard Benchmark", config.data_dir / "conversations.json", config, args.verbose, args.live, judge)
    if args.only != "standard":
        run_suite("Long-Context Stress Benchmark", config.data_dir / "advanced_long_context.json", config,
                  args.verbose, args.live, judge)


if __name__ == "__main__":
    main()
