from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import answer_from_facts, estimate_tokens, extract_profile_updates, is_recall_request
from model_provider import build_chat_model, invoke_with_usage, message_text

BASELINE_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI tiếng Việt. Bạn chỉ nhớ những gì có trong cuộc hội thoại hiện tại; "
    "không có bộ nhớ dài hạn giữa các phiên."
)


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Agent A: within-session memory only.

    - Keeps the full message list per `thread_id` (no compaction, so prompt cost grows linearly per turn).
    - No persistent `User.md`.
    - A new thread starts from zero, so long-term facts are forgotten.
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.langchain_agent = None
        if not force_offline and self.config.live_enabled:
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(thread_id, message)
        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self._session(thread_id).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def memory_file_size(self, user_id: str) -> int:
        # Baseline has no persistent memory file.
        return 0

    def _session(self, thread_id: str) -> SessionState:
        return self.sessions.setdefault(thread_id, SessionState())

    def _prompt_tokens(self, session: SessionState) -> int:
        """Baseline sends system prompt + the whole thread history every turn."""

        return estimate_tokens(BASELINE_SYSTEM_PROMPT) + sum(estimate_tokens(m["content"]) for m in session.messages)

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        prompt_tokens = self._prompt_tokens(session)

        response = self._offline_response(session, message)

        session.messages.append({"role": "assistant", "content": response})
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += estimate_tokens(message) + estimate_tokens(response)
        return {
            "response": response,
            "prompt_tokens": prompt_tokens,
            "agent_tokens": session.token_usage,
            "compactions": 0,
        }

    def _offline_response(self, session: SessionState, message: str) -> str:
        if not is_recall_request(message):
            return "Đã ghi nhận."
        # Short-term memory only: facts are re-read from *this* thread's history.
        facts: dict[str, str] = {}
        interests: list[str] = []
        for past in session.messages[:-1]:
            if past["role"] != "user":
                continue
            updates = extract_profile_updates(past["content"])
            interests += [i.strip() for i in updates.pop("interests", "").split(";") if i.strip()]
            facts.update(updates)
        answer = answer_from_facts(message, facts, list(dict.fromkeys(interests)))
        return answer or "Mình chưa có thông tin này trong cuộc trò chuyện hiện tại."

    # -- live mode -------------------------------------------------------------

    def _reply_live(self, thread_id: str, message: str) -> dict[str, Any]:
        session = self._session(thread_id)
        session.messages.append({"role": "user", "content": message})
        result, input_tokens, output_tokens = invoke_with_usage(
            self.langchain_agent,
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        response = message_text(result["messages"][-1])
        prompt_tokens = input_tokens or self._prompt_tokens(session)
        session.messages.append({"role": "assistant", "content": response})
        session.prompt_tokens_processed += prompt_tokens
        session.token_usage += estimate_tokens(message) + (output_tokens or estimate_tokens(response))
        return {"response": response, "prompt_tokens": prompt_tokens, "agent_tokens": session.token_usage, "compactions": 0}

    def _maybe_build_langchain_agent(self):
        """`create_agent` + `InMemorySaver` (thread-scoped memory only). None if deps are missing."""

        try:
            from langchain.agents import create_agent
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None
        return create_agent(
            model=build_chat_model(self.config.model),
            tools=[],
            system_prompt=BASELINE_SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )
