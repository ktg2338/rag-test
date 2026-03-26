import json
import logging
import re
from typing import List, Tuple

from openai import AzureOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

_client = AzureOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

EXTRACTION_PROMPT = """\
Extract all entity-relation-entity triples from the following text.
Output ONLY a JSON array. Each element must have: {{"subject": "...", "relation": "...", "object": "..."}}

Rules:
- Normalize entity names: capitalize proper nouns, use canonical forms
- Relations should be short verb phrases (e.g., "is CEO of", "located in", "developed by")
- Merge synonyms into a single canonical entity name
- Maximum {max_triples} triples
- If no meaningful triples can be extracted, return an empty array []

Text:
{text}
"""


def extract_triples(text: str) -> List[Tuple[str, str, str]]:
    """텍스트에서 (subject, relation, object) 트리플을 LLM으로 추출"""
    prompt = EXTRACTION_PROMPT.format(
        text=text,
        max_triples=settings.GRAPH_MAX_TRIPLES_PER_CHUNK,
    )

    try:
        resp = _client.chat.completions.create(
            model=settings.AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        return _parse_triples(content)
    except Exception as e:
        logger.error("Triple extraction failed: %s", e)
        return []


def extract_triples_batch(
    chunks: List[str], chunk_ids: List[str] | None = None
) -> List[Tuple[List[Tuple[str, str, str]], str | None]]:
    """청크 리스트에서 배치로 트리플 추출. [(triples, chunk_id), ...] 반환"""
    results = []
    for i, chunk in enumerate(chunks):
        chunk_id = chunk_ids[i] if chunk_ids else None
        triples = extract_triples(chunk)
        results.append((triples, chunk_id))
        logger.info(
            "Chunk %d/%d: extracted %d triples", i + 1, len(chunks), len(triples)
        )
    return results


def _parse_triples(content: str) -> List[Tuple[str, str, str]]:
    """LLM 응답에서 트리플 파싱 (JSON 파싱 + fallback regex)"""
    # JSON 블록 추출 (```json ... ``` 감싸진 경우 대응)
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if json_match:
        content = json_match.group(1).strip()

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [
                (item["subject"], item["relation"], item["object"])
                for item in data
                if isinstance(item, dict)
                and all(k in item for k in ("subject", "relation", "object"))
            ]
    except json.JSONDecodeError:
        pass

    # Fallback: regex로 개별 JSON 객체 추출
    triples = []
    pattern = r'"subject"\s*:\s*"([^"]+)"\s*,\s*"relation"\s*:\s*"([^"]+)"\s*,\s*"object"\s*:\s*"([^"]+)"'
    for match in re.finditer(pattern, content):
        triples.append((match.group(1), match.group(2), match.group(3)))
    return triples
