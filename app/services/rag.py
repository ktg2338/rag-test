from typing import List, Tuple
from uuid import uuid4
from app.services.retriever import retrieve
from app.services.llm import generate_answer
from app.services.memory import memory


def answer_question(
    question: str, top_k: int | None = None, conversation_id: str | None = None
) -> Tuple[str, List[str], str]:
    if conversation_id is None:
        conversation_id = str(uuid4())
    history = memory.get(conversation_id)
    contexts, _metas = retrieve(question, top_k=top_k)
    answer = generate_answer(question, contexts, history)
    memory.append(conversation_id, "user", question)
    memory.append(conversation_id, "assistant", answer)
    return answer, contexts, conversation_id
