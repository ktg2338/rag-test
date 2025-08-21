from typing import List, Tuple
from app.services.retriever import retrieve
from app.services.llm import generate_answer


def answer_question(question: str, top_k: int | None = None) -> Tuple[str, List[str]]:
    contexts, _metas = retrieve(question, top_k=top_k)
    answer = generate_answer(question, contexts)
    return answer, contexts
