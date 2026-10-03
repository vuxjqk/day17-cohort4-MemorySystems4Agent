from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 characters per token)."""

    text = (text or "").strip()
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


# ---------------------------------------------------------------------------
# Profile facts (structured entity extraction + confidence)
# ---------------------------------------------------------------------------

FACT_LABELS = {
    "name": "Tên",
    "location": "Nơi ở hiện tại",
    "profession": "Nghề nghiệp hiện tại",
    "response_style": "Style trả lời",
    "favorite_drink": "Đồ uống yêu thích",
    "favorite_food": "Món ăn yêu thích",
    "pet": "Thú cưng",
}

DEFAULT_CONFIDENCE_THRESHOLD = 0.6


@dataclass
class FactCandidate:
    key: str
    value: str
    confidence: float
    position: int = 0  # char offset in the message; later mentions win on conflict


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_QUESTION_HINTS = re.compile(
    r"\?|\blà gì\b|\btên gì\b|\bở đâu\b|\blàm gì\b|\bnhư thế nào\b|\bthế nào\b|\bbao nhiêu\b|\bcon gì\b"
)
_NOISE_HINTS = re.compile(r"đùa|giả sử|ví dụ cũ|chỉ là nơi|thông tin cũ|hay là chuyển")
_HEDGE_HINTS = re.compile(r"^nếu\b|có lẽ|hình như|không chắc|chắc là|biết đâu")
_STALE_HINTS = re.compile(r"lúc đầu|trước đó|trước đây|ban đầu")
_NEGATION_BEFORE = re.compile(r"(không còn|không phải|chứ không|không|đừng|chưa)\b[^,;:]*$")

_WORDS = r"(\w+(?:[ \t]+\w+){0,3})"
_NAME_PATTERNS = [
    re.compile(r"(?:mình|tôi|em)\s+tên\s+(?:là\s+)?" + _WORDS, re.IGNORECASE),
    re.compile(r"tên\s+(?:của\s+)?(?:mình|tôi|em)\s+là\s+" + _WORDS, re.IGNORECASE),
    re.compile(r"(?:^|[:,]\s*)tên\s+(?:là\s+)?" + _WORDS, re.IGNORECASE),
]
_LOCATION_PATTERNS = [
    (re.compile(r"cập nhật từ\s+[^,.;]+?\s+sang\s+" + _WORDS), 0.95),
    (re.compile(r"nơi ở(?: hiện tại)?(?: vẫn)? là\s+" + _WORDS), 0.9),
    (re.compile(r"\bở\s+" + _WORDS), 0.8),
]
_PROFESSION_PATTERN = re.compile(
    r"\b([A-Za-z][A-Za-z0-9\-]*\s+(?:engineer|developer|manager|scientist|designer|analyst|researcher))\b"
)
_PROFESSION_CONTEXT = re.compile(r"\blàm\b|nghề|chuyển sang|\blà\b")
_DRINK_PATTERNS = [
    (re.compile(r"đồ uống yêu thích(?: của mình)?(?: vẫn)? là\s+([^.,;!?]+)", re.IGNORECASE), 0.9),
    (re.compile(r"(?:mình|tôi)\s+(?:vẫn\s+|hay\s+)?uống\s+(.+?)(?=\s+(?:như cũ|nhưng|mỗi ngày)|[.,;!?]|$)", re.IGNORECASE), 0.7),
]
_FOOD_PATTERN = re.compile(r"món (?:ăn )?yêu thích(?: của mình)?(?: vẫn)? là\s+([^.,;!?]+)", re.IGNORECASE)
_PET_PATTERN = re.compile(r"\bnuôi\s+(?:một\s+|1\s+)?(?:bé\s+|con\s+|chú\s+)?([^.,;!?]+)", re.IGNORECASE)
_INTEREST_PATTERN = re.compile(
    r"(?:mình|tôi)\s+(?:vẫn\s+|còn\s+|cũng\s+|đang\s+)*(?:thích|quan tâm(?: nhiều)?(?: đến| tới)?)\s+(.+)",
    re.IGNORECASE,
)
_INTEREST_STOP_PREFIX = re.compile(r"^(cách|kiểu|tin|hơn|câu|style|trả lời|giải thích|bạn|nó|đây|mình)\b", re.IGNORECASE)
_STYLE_TOPIC = re.compile(r"trả lời|giải thích|style|trình bày")
_STYLE_INTENT = re.compile(r"muốn|thích|hãy|ưu tiên|giữ|theo dạng|\bnên\b|lan man")
_STYLE_COMPONENTS = [
    (re.compile(r"ngắn|\bgọn\b|lan man"), "ngắn gọn"),
    (re.compile(r"\b(?:3|ba) bullet"), "3 bullet"),
    (re.compile(r"bullet"), "bullet"),
    (re.compile(r"cấu trúc"), "có cấu trúc"),
    (re.compile(r"ví dụ thực tế"), "có ví dụ thực tế"),
    (re.compile(r"thực chiến"), "có ví dụ thực chiến"),
    (re.compile(r"trade-off"), "nhấn trade-off"),
]


def _leading_capitalized(words: str) -> str:
    """Keep the leading capitalized tokens: `Đà Nẵng vài tháng` -> `Đà Nẵng`."""

    kept = []
    for token in words.split():
        if token[:1].isupper():
            kept.append(token)
        else:
            break
    return " ".join(kept)


def is_question(text: str) -> bool:
    return bool(_QUESTION_HINTS.search(text.lower()))


def _split_sentences(message: str) -> list[tuple[int, str]]:
    out, offset = [], 0
    for part in _SENTENCE_SPLIT.split(message.strip()):
        idx = message.find(part, offset)
        offset = idx + len(part)
        if part.strip():
            out.append((idx, part.strip()))
    return out


def _is_negated(sentence: str, start: int) -> bool:
    return bool(_NEGATION_BEFORE.search(sentence[max(0, start - 24):start].lower()))


def _clean_value(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" .,:;!?\"'")
    value = re.sub(r"\s+(?:nhé|nha|ạ|đó|nữa|như cũ)$", "", value)
    return value.strip()


def extract_profile_candidates(message: str) -> list[FactCandidate]:
    """Structured entity extraction with a confidence score per fact.

    Guardrails (bonus):
    - question sentences never write facts ("Mình tên gì?")
    - noise/joke sentences are ignored ("chỉ là câu đùa", "chỉ là nơi đi họp")
    - negated mentions are ignored ("không còn làm backend engineer")
    - hedged / stale mentions get a lower confidence ("Nếu...", "Lúc đầu mình nói...")
    """

    candidates: list[FactCandidate] = []
    for offset, sentence in _split_sentences(message):
        lower = sentence.lower()
        if is_question(sentence):
            continue
        noisy = bool(_NOISE_HINTS.search(lower))
        penalty = 0.3 if _HEDGE_HINTS.search(lower) else 0.0

        for pattern in _NAME_PATTERNS:
            for match in pattern.finditer(sentence):
                name = _leading_capitalized(match.group(1))
                if name:
                    candidates.append(FactCandidate("name", name, 0.9 - penalty, offset + match.start(1)))

        if not noisy:
            for pattern, confidence in _LOCATION_PATTERNS:
                for match in pattern.finditer(sentence):
                    place = _leading_capitalized(match.group(1))
                    if not place or _is_negated(sentence, match.start()):
                        continue
                    conf = confidence - penalty
                    if _STALE_HINTS.search(lower[max(0, match.start() - 40):match.start()]):
                        conf -= 0.4
                    candidates.append(FactCandidate("location", place, conf, offset + match.start(1)))

            if _PROFESSION_CONTEXT.search(lower):
                for match in _PROFESSION_PATTERN.finditer(sentence):
                    if _is_negated(sentence, match.start()):
                        continue
                    candidates.append(
                        FactCandidate("profession", match.group(1), 0.85 - penalty, offset + match.start(1))
                    )

        for pattern, confidence in _DRINK_PATTERNS:
            for match in pattern.finditer(sentence):
                value = _clean_value(match.group(1))
                if value and len(value.split()) <= 5:
                    candidates.append(FactCandidate("favorite_drink", value, confidence - penalty, offset + match.start(1)))

        match = _FOOD_PATTERN.search(sentence)
        if match:
            candidates.append(FactCandidate("favorite_food", _clean_value(match.group(1)), 0.9 - penalty, offset + match.start(1)))

        match = _PET_PATTERN.search(sentence)
        if match:
            value = _clean_value(match.group(1))
            if value and len(value.split()) <= 5:
                candidates.append(FactCandidate("pet", value, 0.85 - penalty, offset + match.start(1)))

        if _STYLE_TOPIC.search(lower) and _STYLE_INTENT.search(lower):
            parts: list[str] = []
            for pattern, label in _STYLE_COMPONENTS:
                if pattern.search(lower) and not (label == "bullet" and "3 bullet" in parts):
                    parts.append(label)
            if parts:
                candidates.append(FactCandidate("response_style", ", ".join(parts), 0.8 - penalty, offset))

        match = _INTEREST_PATTERN.search(sentence)
        if match:
            items = []
            raw = re.split(r"\s+(?:vì|để|hơn là|khi|nhưng)\s+", match.group(1))[0]
            for item in re.split(r",\s*|\s+và\s+", raw):
                item = _clean_value(item)
                if (
                    item
                    and len(item.split()) <= 4
                    and not _INTEREST_STOP_PREFIX.search(item)
                    and not re.search(r"\b(này|đó)\b", item)
                ):
                    items.append(item)
            if items:
                candidates.append(FactCandidate("interests", "; ".join(items), 0.75 - penalty, offset))

    return candidates


def extract_profile_updates(message: str, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> dict[str, str]:
    """Convert raw user text into stable profile facts (only confident ones).

    When the same key appears several times, the latest confident mention wins:
    "Lúc đầu mình nói ở Huế, nhưng thực ra ... ở Đà Nẵng" -> location = Đà Nẵng.
    """

    updates: dict[str, str] = {}
    for candidate in sorted(extract_profile_candidates(message), key=lambda c: c.position):
        if candidate.confidence < threshold:
            continue
        if candidate.key == "interests" and "interests" in updates:
            updates["interests"] += "; " + candidate.value
        elif candidate.key == "response_style" and "response_style" in updates:
            updates["response_style"] = merge_style(updates["response_style"], candidate.value)
        else:
            updates[candidate.key] = candidate.value
    return updates


def merge_style(old: str, new: str) -> str:
    parts = [p.strip() for p in old.split(",") if p.strip()]
    for part in (p.strip() for p in new.split(",")):
        if part and part not in parts:
            parts.append(part)
    if "3 bullet" in parts and "bullet" in parts:
        parts.remove("bullet")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Persistent memory: User.md
# ---------------------------------------------------------------------------

_FACT_LINE = re.compile(r"^- (\w+): (.+?) _\(confidence ([\d.]+), turn (\d+)\)_$")
_INTEREST_LINE = re.compile(r"^- (.+?) _\(mentions (\d+), last turn (\d+)\)_$")
_META_LINE = re.compile(r"^- turns_seen: (\d+)$")


@dataclass
class ProfileData:
    facts: dict[str, dict[str, object]] = field(default_factory=dict)
    interests: dict[str, dict[str, int]] = field(default_factory=dict)
    corrections: list[str] = field(default_factory=list)
    turns_seen: int = 0


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md` (one markdown file per user).

    Layout: `<root_dir>/<user_slug>/User.md` with sections Profile / Interests /
    Corrections / Meta. Markdown keeps it human-readable and easy to diff.
    """

    root_dir: Path
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    interest_decay: float = 0.97
    max_interests: int = 8
    max_corrections: int = 5

    # -- raw file operations -------------------------------------------------

    def path_for(self, user_id: str) -> Path:
        slug = re.sub(r"[^\w\-]+", "_", user_id.strip().lower()).strip("_") or "anonymous"
        return Path(self.root_dir) / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return f"# User.md\n\n<!-- user_id: {user_id} -->\n\n## Profile\n_(chưa có thông tin)_\n"

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        content = self.read_text(user_id)
        if not search_text or search_text not in content:
            return False
        self.write_text(user_id, content.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    # -- structured helpers --------------------------------------------------

    def load(self, user_id: str) -> ProfileData:
        data = ProfileData()
        section = None
        for line in self.read_text(user_id).splitlines():
            if line.startswith("## "):
                section = line[3:].strip().lower()
                continue
            if section == "profile" and (m := _FACT_LINE.match(line)):
                data.facts[m.group(1)] = {"value": m.group(2), "confidence": float(m.group(3)), "turn": int(m.group(4))}
            elif section == "interests" and (m := _INTEREST_LINE.match(line)):
                data.interests[m.group(1)] = {"mentions": int(m.group(2)), "last": int(m.group(3))}
            elif section == "corrections" and line.startswith("- "):
                data.corrections.append(line[2:])
            elif section == "meta" and (m := _META_LINE.match(line)):
                data.turns_seen = int(m.group(1))
        return data

    def save(self, user_id: str, data: ProfileData) -> Path:
        lines = ["# User.md", "", f"<!-- user_id: {user_id} -->", "", "## Profile"]
        ordered = [k for k in FACT_LABELS if k in data.facts] + [k for k in data.facts if k not in FACT_LABELS]
        for key in ordered:
            fact = data.facts[key]
            lines.append(f"- {key}: {fact['value']} _(confidence {float(fact['confidence']):.2f}, turn {fact['turn']})_")
        if data.interests:
            lines += ["", "## Interests"]
            for name, info in self.ranked_interests(data):
                lines.append(f"- {name} _(mentions {info['mentions']}, last turn {info['last']})_")
        if data.corrections:
            lines += ["", "## Corrections"] + [f"- {c}" for c in data.corrections[-self.max_corrections:]]
        lines += ["", "## Meta", f"- turns_seen: {data.turns_seen}", ""]
        return self.write_text(user_id, "\n".join(lines))

    def ranked_interests(self, data: ProfileData) -> list[tuple[str, dict[str, int]]]:
        """Memory decay: score = mentions * decay ** (turns since last mention)."""

        def score(item: tuple[str, dict[str, int]]) -> float:
            info = item[1]
            return info["mentions"] * self.interest_decay ** max(0, data.turns_seen - info["last"])

        return sorted(data.interests.items(), key=score, reverse=True)[: self.max_interests]

    def facts(self, user_id: str) -> dict[str, str]:
        return {k: str(v["value"]) for k, v in self.load(user_id).facts.items()}

    def interests(self, user_id: str, limit: int = 4) -> list[str]:
        return [name for name, _ in self.ranked_interests(self.load(user_id))[:limit]]

    def upsert_fact(self, user_id: str, key: str, value: str, confidence: float = 1.0) -> bool:
        return bool(self.apply_updates(user_id, {key: value}, confidence=confidence, count_turn=False))

    def apply_updates(
        self,
        user_id: str,
        updates: dict[str, str],
        confidence: float = 0.8,
        count_turn: bool = True,
        confidences: dict[str, float] | None = None,
    ) -> dict[str, str]:
        """Merge extracted facts into User.md with conflict handling.

        - a new value for a single-valued fact replaces the old one (correction)
          and the change is logged under `## Corrections` (capped)
        - response_style merges components instead of overwriting
        - interests accumulate mention counts (used by decay ranking)
        Returns the facts that actually changed.
        """

        data = self.load(user_id)
        if count_turn:
            data.turns_seen += 1
        turn = data.turns_seen
        changed: dict[str, str] = {}

        for key, value in updates.items():
            if key == "interests":
                for item in [v.strip() for v in value.split(";") if v.strip()]:
                    match = next((k for k in data.interests if k.lower() == item.lower()), item)
                    info = data.interests.setdefault(match, {"mentions": 0, "last": turn})
                    info["mentions"] += 1
                    info["last"] = turn
                continue
            old = data.facts.get(key)
            new_value = merge_style(str(old["value"]), value) if key == "response_style" and old else value
            if old and str(old["value"]) == new_value:
                old["turn"] = turn
                continue
            if old and key != "response_style":
                data.corrections.append(f"turn {turn}: {key} `{old['value']}` -> `{new_value}`")
            conf = (confidences or {}).get(key, confidence)
            data.facts[key] = {"value": new_value, "confidence": conf, "turn": turn}
            changed[key] = new_value

        if len(data.interests) > self.max_interests * 2:
            data.interests = dict(self.ranked_interests(data))
        if updates or count_turn:
            self.save(user_id, data)
        return changed

    def ingest_message(self, user_id: str, message: str) -> dict[str, str]:
        """Extract confident facts from one user message and persist them."""

        updates = extract_profile_updates(message, threshold=self.confidence_threshold)
        confidences: dict[str, float] = {}
        for candidate in extract_profile_candidates(message):
            if candidate.key in updates and candidate.confidence >= self.confidence_threshold:
                confidences[candidate.key] = max(confidences.get(candidate.key, 0.0), candidate.confidence)
        return self.apply_updates(user_id, updates, confidences=confidences)


# ---------------------------------------------------------------------------
# Answering from memory (shared by both agents' offline paths)
# ---------------------------------------------------------------------------

_QUESTION_KEYS = [
    ("name", re.compile(r"tên|là ai|tóm tắt|mô tả")),
    ("profession", re.compile(r"nghề|làm gì|công việc|là ai|tóm tắt|mô tả")),
    ("location", re.compile(r"ở đâu|nơi ở|\bở\b|sống")),
    ("response_style", re.compile(r"style|trả lời|kiểu")),
    ("favorite_drink", re.compile(r"đồ uống|uống")),
    ("favorite_food", re.compile(r"món")),
    ("pet", re.compile(r"nuôi|con gì|thú cưng")),
    ("interests", re.compile(r"quan tâm|sở thích|thích gì|là ai|tóm tắt|mô tả")),
]


def requested_fact_keys(question: str) -> list[str]:
    lower = question.lower()
    return [key for key, pattern in _QUESTION_KEYS if pattern.search(lower)]


def is_recall_request(message: str) -> bool:
    lower = message.lower()
    return is_question(message) or bool(
        re.search(r"^nhắc lại\b|nhắc lại giúp|tóm tắt ngắn về mình|bạn có biết|bạn thử nhớ lại|nhớ lại xem", lower)
    )


def answer_from_facts(question: str, facts: dict[str, str], interests: list[str], bullets: bool = False) -> str | None:
    """Build a short answer for a recall question; None if no fact is relevant/known."""

    keys = requested_fact_keys(question)
    lines = []
    for key in keys:
        if key == "interests":
            if interests:
                lines.append(f"Mối quan tâm chính: {', '.join(interests[:4])}")
        elif key in facts:
            lines.append(f"{FACT_LABELS[key]}: {facts[key]}")
    if not lines:
        return None
    if bullets:
        return "\n".join(f"- {line}" for line in lines)
    return "; ".join(lines) + "."


# ---------------------------------------------------------------------------
# Compact memory
# ---------------------------------------------------------------------------


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Heuristic summary of older messages: first sentence of each user turn, clipped.

    Assistant turns are dropped (they mostly echo the user). Replace with an
    LLM summary in live mode (SummarizationMiddleware) if desired.
    """

    items = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "").strip()
        if not content:
            continue
        first = _SENTENCE_SPLIT.split(content)[0]
        if len(first) > 110:
            first = first[:107].rstrip() + "..."
        items.append(f"- {first}")
    return "\n".join(items[-max_items:])


@dataclass
class CompactMemoryManager:
    """Short-term memory per thread with automatic compaction.

    Recent `keep_messages` stay verbatim; when summary + messages exceed
    `threshold_tokens`, older messages are folded into a bounded summary.
    """

    threshold_tokens: int
    keep_messages: int
    max_summary_items: int = 8
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        ctx = self.context(thread_id)
        ctx["messages"].append({"role": role, "content": content})
        if self.context_tokens(thread_id) > self.threshold_tokens and len(ctx["messages"]) > self.keep_messages:
            self.compact(thread_id)

    def compact(self, thread_id: str) -> None:
        ctx = self.context(thread_id)
        messages: list[dict[str, str]] = ctx["messages"]
        cut = len(messages) - self.keep_messages
        if cut <= 0:
            return
        old, ctx["messages"] = messages[:cut], messages[cut:]
        previous = [line for line in str(ctx["summary"]).splitlines() if line.strip()]
        fresh = summarize_messages(old, max_items=self.max_summary_items).splitlines()
        ctx["summary"] = "\n".join((previous + fresh)[-self.max_summary_items:])
        ctx["compactions"] = int(ctx["compactions"]) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(thread_id, {"messages": [], "summary": "", "compactions": 0})

    def context_tokens(self, thread_id: str) -> int:
        ctx = self.context(thread_id)
        return estimate_tokens(str(ctx["summary"])) + sum(estimate_tokens(m["content"]) for m in ctx["messages"])

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id)["compactions"])
