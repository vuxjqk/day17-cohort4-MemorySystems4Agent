from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    FACT_LABELS,
    CompactMemoryManager,
    UserProfileStore,
    answer_from_facts,
    estimate_tokens,
    is_recall_request,
)
from model_provider import build_chat_model, invoke_with_usage, message_text

try:  # live-mode deps; module level so tool/prompt type hints can be resolved
    from langchain.agents.middleware import ModelRequest
    from langchain.tools import ToolRuntime
except ImportError:  # offline mode works without LangChain
    ModelRequest = ToolRuntime = None

ADVANCED_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI tiếng Việt có bộ nhớ dài hạn. Hồ sơ người dùng (User.md) bên dưới là nguồn sự thật "
    "cho các fact ổn định; nếu có đính chính thì luôn dùng giá trị mới nhất. Tuân theo style trả lời trong hồ sơ."
)


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Agent B: three memory layers.

    1. short-term: recent messages of the current thread (verbatim)
    2. persistent: `User.md` per user, survives across threads/sessions
    3. compact: older thread messages are folded into a bounded summary
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(
            self.config.state_dir / "profiles",
            confidence_threshold=self.config.profile_confidence_threshold,
            interest_decay=self.config.interest_decay,
            max_interests=self.config.max_interests,
        )
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.live_history_size: dict[str, int] = {}
        self.live_compactions: dict[str, int] = {}
        self.langchain_agent = None
        if not force_offline and self.config.live_enabled:
            self.langchain_agent = self._maybe_build_langchain_agent()

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        if self.langchain_agent is not None:
            return self._reply_live(user_id, thread_id, message)
        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        if self.langchain_agent is not None:
            return self.live_compactions.get(thread_id, 0)
        return self.compact_memory.compaction_count(thread_id)

    def _remember(self, user_id: str, thread_id: str, message: str) -> tuple[dict[str, str], int]:
        """Steps 1-4 of a turn: persist facts, append to short-term memory, measure prompt load."""

        changed = self.profile_store.ingest_message(user_id, message)
        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens
        return changed, prompt_tokens

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        changed, prompt_tokens = self._remember(user_id, thread_id, message)

        response = self._offline_response(user_id, thread_id, message, changed)

        self.compact_memory.append(thread_id, "assistant", response)
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0) + estimate_tokens(message) + estimate_tokens(response)
        )
        return {
            "response": response,
            "prompt_tokens": prompt_tokens,
            "agent_tokens": self.thread_tokens[thread_id],
            "profile_updates": changed,
            "compactions": self.compaction_count(thread_id),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """System prompt + User.md + compact summary + recent kept messages."""

        return (
            estimate_tokens(ADVANCED_SYSTEM_PROMPT)
            + estimate_tokens(self.profile_store.read_text(user_id))
            + self.compact_memory.context_tokens(thread_id)
        )

    def _offline_response(
        self, user_id: str, thread_id: str, message: str, changed: dict[str, str] | None = None
    ) -> str:
        """Deterministic answer that reads persisted memory first, then the thread summary."""

        facts = self.profile_store.facts(user_id)
        bullets = "bullet" in facts.get("response_style", "")

        if is_recall_request(message):
            answer = answer_from_facts(message, facts, self.profile_store.interests(user_id), bullets=bullets)
            if answer:
                return answer
            summary = str(self.compact_memory.context(thread_id)["summary"])
            if summary:
                return "Mình chưa có fact này trong User.md. Tóm tắt phần trước của thread:\n" + summary
            return "Mình chưa có thông tin này trong User.md."

        if changed:
            saved = ", ".join(f"{FACT_LABELS.get(k, k)} = {v}" for k, v in changed.items())
            return f"Đã ghi nhận và cập nhật User.md: {saved}."
        return "Đã ghi nhận."

    # -- live mode -------------------------------------------------------------

    def _reply_live(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        changed, estimated_prompt = self._remember(user_id, thread_id, message)
        context = AgentContext(user_id=user_id, memory_path=str(self.profile_store.path_for(user_id)))
        result, input_tokens, output_tokens = invoke_with_usage(
            self.langchain_agent,
            {"messages": [{"role": "user", "content": message}]},
            config={"configurable": {"thread_id": thread_id}},
            context=context,
        )
        response = message_text(result["messages"][-1])
        # SummarizationMiddleware replaces old messages with one summary, so the
        # checkpointed history shrinks: that is a real compaction.
        size = len(result["messages"])
        if size < self.live_history_size.get(thread_id, 0):
            self.live_compactions[thread_id] = self.live_compactions.get(thread_id, 0) + 1
        self.live_history_size[thread_id] = size
        if input_tokens:
            # Replace the estimate with the provider's real number.
            self.thread_prompt_tokens[thread_id] += input_tokens - estimated_prompt
        self.compact_memory.append(thread_id, "assistant", response)
        self.thread_tokens[thread_id] = (
            self.thread_tokens.get(thread_id, 0)
            + estimate_tokens(message)
            + (output_tokens or estimate_tokens(response))
        )
        return {
            "response": response,
            "prompt_tokens": input_tokens or estimated_prompt,
            "agent_tokens": self.thread_tokens[thread_id],
            "profile_updates": changed,
            "compactions": self.compaction_count(thread_id),
        }

    def _maybe_build_langchain_agent(self):
        """Live agent: InMemorySaver + User.md tools + dynamic profile prompt + summarization.

        Returns None when LangChain/LangGraph are not installed (offline path is used instead).
        """

        try:
            from langchain.agents import create_agent
            from langchain.agents.middleware import SummarizationMiddleware, dynamic_prompt
            from langchain.tools import tool
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            return None

        store = self.profile_store
        model = build_chat_model(self.config.model)

        @tool
        def read_user_memory(runtime: ToolRuntime[AgentContext]) -> str:
            """Đọc toàn bộ User.md của người dùng hiện tại."""

            return store.read_text(runtime.context.user_id)

        @tool
        def save_user_fact(key: str, value: str, runtime: ToolRuntime[AgentContext]) -> str:
            """Lưu/cập nhật một fact ổn định (name, location, profession, response_style,
            favorite_drink, favorite_food, pet) vào User.md. Chỉ dùng khi người dùng khẳng định rõ."""

            store.upsert_fact(runtime.context.user_id, key, value, confidence=0.8)
            return f"saved {key}={value}"

        @tool
        def edit_user_memory(search_text: str, replacement: str, runtime: ToolRuntime[AgentContext]) -> str:
            """Sửa một đoạn text trong User.md (thay thế lần xuất hiện đầu tiên)."""

            ok = store.edit_text(runtime.context.user_id, search_text, replacement)
            return "edited" if ok else "search_text not found"

        @dynamic_prompt
        def profile_prompt(request: ModelRequest) -> str:
            user_md = store.read_text(request.runtime.context.user_id)
            return f"{ADVANCED_SYSTEM_PROMPT}\n\n<user_md>\n{user_md}\n</user_md>"

        try:
            summarizer = SummarizationMiddleware(
                model=model,
                trigger=("tokens", self.config.compact_threshold_tokens),
                keep=("messages", self.config.compact_keep_messages),
            )
        except TypeError:  # older langchain 1.0.x signature
            summarizer = SummarizationMiddleware(
                model=model,
                max_tokens_before_summary=self.config.compact_threshold_tokens,
                messages_to_keep=self.config.compact_keep_messages,
            )

        return create_agent(
            model=model,
            tools=[read_user_memory, save_user_fact, edit_user_memory],
            middleware=[profile_prompt, summarizer],
            context_schema=AgentContext,
            checkpointer=InMemorySaver(),
        )
