from __future__ import annotations
from typing import Dict, List, TypedDict


class Message(TypedDict):
    role: str
    content: str


class ConversationMemory:
    """Simple in-memory store for chat histories."""

    def __init__(self) -> None:
        self._store: Dict[str, List[Message]] = {}

    def get(self, conversation_id: str) -> List[Message]:
        return self._store.get(conversation_id, [])

    def append(self, conversation_id: str, role: str, content: str) -> None:
        self._store.setdefault(conversation_id, []).append({"role": role, "content": content})


memory = ConversationMemory()
