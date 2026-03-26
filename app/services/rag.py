from typing import List, Tuple
from uuid import uuid4

from app.services.agent import rag_agent
from app.services.memory import memory


def answer_question(
    question: str, top_k: int | None = None, conversation_id: str | None = None
) -> Tuple[str, List[str], str, List[str]]:
    if conversation_id is None:
        conversation_id = str(uuid4())

    history = memory.get(conversation_id)

    # LangGraph Agent 실행
    initial_state = {
        "question": question,
        "conversation_id": conversation_id,
        "history": history,
        "retry_count": 0,
        "steps": [],
    }
    if top_k is not None:
        initial_state["top_k"] = top_k

    result = rag_agent.invoke(initial_state)

    answer = result.get("generation", "")
    contexts = result.get("filtered_documents") or result.get("documents", [])
    steps = result.get("steps", [])

    memory.append(conversation_id, "user", question)
    memory.append(conversation_id, "assistant", answer)

    return answer, contexts, conversation_id, steps
