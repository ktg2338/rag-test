from typing import List
from openai import OpenAI
from app.core.config import settings

_client = OpenAI(api_key=settings.OPENAI_API_KEY)


def generate_answer(question: str, contexts: List[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts) if contexts else "N/A"
    system = (
        "You are a helpful assistant that answers strictly based on the provided context. "
        "If the answer is not contained in the context, say you don't know."
    )
    user = (
        f"# Question\n{question}\n\n"
        f"# Context\n{context_block}\n\n"
        "Answer in Korean. Include brief citations like [#1], [#2] referring to the order of context chunks if useful."
    )

    resp = _client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
