"""
공시 데이터를 RAG 인덱스에 적재할 (texts, metadatas) 형태로 변환.
"""

from typing import Any

from app.services.chunker import chunk_text


def build_records(
    disclosure: dict[str, Any],
    body: str | None,
    stock: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    하나의 공시 → 청크 리스트 + 메타데이터 리스트.

    body가 None이면 제목/메타만으로 단일 레코드 생성 (최소한의 검색 가능성 보장).
    """
    rcept_no = disclosure["rcept_no"]
    rcept_dt = disclosure.get("rcept_dt", "")
    report_nm = disclosure.get("report_nm", "")
    flr_nm = disclosure.get("flr_nm", "")
    corp_name = disclosure.get("corp_name") or stock.get("name", "")
    corp_code = disclosure.get("corp_code", "")

    dart_url = f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"

    base_meta = {
        "source": "dart",
        "ticker": stock.get("ticker", ""),
        "corp_code": corp_code,
        "corp_name": corp_name,
        "market": stock.get("market", ""),
        "report_nm": report_nm,
        "rcept_no": rcept_no,
        "rcept_dt": rcept_dt,
        "filer": flr_nm,
        "url": dart_url,
    }

    # 검색 품질을 위해 본문 앞에 컨텍스트 헤더를 prepend
    header = (
        f"[{corp_name}({stock.get('ticker', '')}) - {report_nm}] "
        f"공시일: {rcept_dt}\n"
    )

    if not body:
        return [header.strip()], [{**base_meta, "chunk": 0, "has_body": False}]

    full_text = header + body
    chunks = chunk_text(full_text)
    metas = [
        {**base_meta, "chunk": i, "has_body": True}
        for i in range(len(chunks))
    ]
    return chunks, metas
