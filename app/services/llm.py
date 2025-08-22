from typing import List, Optional
from openai import AzureOpenAI
from app.core.config import settings
from app.services.memory import Message

_client = AzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

def generate_answer(
    question: str, contexts: List[str], history: Optional[List[Message]] = None
) -> str:
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

    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})

    resp = _client.chat.completions.create(
        model=settings.AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
