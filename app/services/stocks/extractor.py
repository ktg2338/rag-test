"""
DART 공시 본문 추출.

document.xml API는 zip을 반환하며 내부에 1개 이상의 XML 파일이 있다.
공시 종류에 따라 본문이 비어있거나 XBRL일 수 있어, 텍스트가 빈 경우는 None 반환.
"""

import io
import re
import zipfile
from xml.etree import ElementTree as ET

import requests

from app.core.config import settings


_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTILINE_RE = re.compile(r"\n{3,}")


def _xml_to_text(root: ET.Element) -> str:
    """XML 트리에서 텍스트 노드만 모아 정리."""
    parts: list[str] = []
    for elem in root.iter():
        # XBRL 태그(예: ix:nonNumeric)나 일반 태그 모두 .text/.tail 수집
        if elem.text:
            t = elem.text.strip()
            if t:
                parts.append(t)
        if elem.tail:
            t = elem.tail.strip()
            if t:
                parts.append(t)
    text = "\n".join(parts)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTILINE_RE.sub("\n\n", text)
    return text.strip()


def fetch_document_text(rcept_no: str, max_chars: int | None = None) -> str | None:
    """
    rcept_no에 해당하는 공시 본문 텍스트를 반환.

    실패하거나 본문이 비어있으면 None.
    """
    max_chars = max_chars if max_chars is not None else settings.DART_MAX_DOC_CHARS

    url = f"{settings.DART_BASE_URL}/document.xml"
    try:
        resp = requests.get(
            url,
            params={"crtfc_key": settings.DART_API_KEY, "rcept_no": rcept_no},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return None

    # 응답이 zip이 아닌 JSON 에러 메시지인 경우가 있음
    if not resp.content[:2] == b"PK":
        return None

    chunks: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".xml"):
                    continue
                try:
                    with zf.open(name) as f:
                        tree = ET.parse(f)
                    chunks.append(_xml_to_text(tree.getroot()))
                except ET.ParseError:
                    continue
    except zipfile.BadZipFile:
        return None

    text = "\n\n".join(c for c in chunks if c).strip()
    if not text:
        return None
    if len(text) > max_chars:
        text = text[:max_chars]
    return text
