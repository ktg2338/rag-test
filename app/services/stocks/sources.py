"""
국내 주식 데이터 소스 클라이언트.

- 시총 상위 종목 리스트: 하드코딩 (테스트용, KRX 인증 회피)
- DART Open API: corp_code 매핑, 공시 목록 조회
"""

import io
import json
import os
import zipfile
from typing import Any
from xml.etree import ElementTree as ET

import requests

from app.core.config import settings


CORPCODE_CACHE_PATH = os.path.join(settings.DART_STATE_DIR, "dart_corpcode.json")


# ---------------------------------------------------------------------------
# 시총 상위 종목 하드코딩 리스트 (시총 내림차순)
# ---------------------------------------------------------------------------
# 테스트용. 실제 운영에서는 fresh data 소스(KRX 인증, FinanceDataReader 등)로 교체.
# 정확한 실시간 순위가 아니어도 ETL 파이프라인 동작 검증에는 충분.

KOSPI_TOP: list[tuple[str, str]] = [
    ("005930", "삼성전자"),
    ("000660", "SK하이닉스"),
    ("373220", "LG에너지솔루션"),
    ("207940", "삼성바이오로직스"),
    ("005380", "현대차"),
    ("005935", "삼성전자우"),
    ("000270", "기아"),
    ("068270", "셀트리온"),
    ("105560", "KB금융"),
    ("035420", "NAVER"),
    ("005490", "POSCO홀딩스"),
    ("012330", "현대모비스"),
    ("055550", "신한지주"),
    ("035720", "카카오"),
    ("028260", "삼성물산"),
    ("051910", "LG화학"),
    ("006400", "삼성SDI"),
    ("003670", "포스코퓨처엠"),
    ("015760", "한국전력"),
    ("086790", "하나금융지주"),
    ("032830", "삼성생명"),
    ("010130", "고려아연"),
    ("000810", "삼성화재"),
    ("011200", "HMM"),
    ("009150", "삼성전기"),
    ("034730", "SK"),
    ("018260", "삼성에스디에스"),
    ("033780", "KT&G"),
    ("138040", "메리츠금융지주"),
    ("017670", "SK텔레콤"),
]

KOSDAQ_TOP: list[tuple[str, str]] = [
    ("247540", "에코프로비엠"),
    ("086520", "에코프로"),
    ("196170", "알테오젠"),
    ("028300", "HLB"),
    ("022100", "포스코DX"),
    ("357780", "솔브레인"),
    ("293490", "카카오게임즈"),
    ("263750", "펄어비스"),
    ("035900", "JYP Ent."),
    ("058470", "리노공업"),
    ("277810", "레인보우로보틱스"),
    ("095340", "ISC"),
    ("214150", "클래시스"),
    ("066970", "엘앤에프"),
    ("042700", "한미반도체"),
    ("058820", "CMG제약"),
    ("039030", "이오테크닉스"),
    ("067310", "하나마이크론"),
    ("240810", "원익IPS"),
    ("357230", "주성엔지니어링"),
]


def _require_key() -> str:
    key = settings.DART_API_KEY
    if not key:
        raise RuntimeError(
            "DART_API_KEY가 설정되지 않았습니다. .env에 DART_API_KEY=... 를 추가하세요."
        )
    return key


# ---------------------------------------------------------------------------
# 시총 상위 종목 조회
# ---------------------------------------------------------------------------

def get_top_stocks(top_n: int = 10, market: str = "KOSPI") -> list[dict[str, Any]]:
    """시가총액 상위 종목 리스트 (하드코딩된 정적 데이터).

    Returns: [{"ticker": "005930", "name": "삼성전자", "market": "KOSPI"}, ...]
    """
    if market == "KOSPI":
        source = KOSPI_TOP
    elif market == "KOSDAQ":
        source = KOSDAQ_TOP
    else:
        raise ValueError(f"지원하지 않는 시장: {market}")

    return [
        {"ticker": ticker, "name": name, "market": market}
        for ticker, name in source[:top_n]
    ]


# ---------------------------------------------------------------------------
# DART corp_code 매핑 (stock_code → corp_code)
# ---------------------------------------------------------------------------

def _download_corpcode_map() -> dict[str, dict[str, str]]:
    """DART corpCode.xml 다운로드 및 파싱."""
    url = f"{settings.DART_BASE_URL}/corpCode.xml"
    resp = requests.get(url, params={"crtfc_key": _require_key()}, timeout=30)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        with zf.open("CORPCODE.xml") as f:
            tree = ET.parse(f)

    mapping: dict[str, dict[str, str]] = {}
    for node in tree.getroot().findall("list"):
        stock_code = (node.findtext("stock_code") or "").strip()
        corp_code = (node.findtext("corp_code") or "").strip()
        corp_name = (node.findtext("corp_name") or "").strip()
        if stock_code and corp_code:
            mapping[stock_code] = {"corp_code": corp_code, "corp_name": corp_name}
    return mapping


def get_corpcode_map(force_refresh: bool = False) -> dict[str, dict[str, str]]:
    """stock_code → {corp_code, corp_name} 매핑. 로컬에 캐시."""
    if not force_refresh and os.path.exists(CORPCODE_CACHE_PATH):
        with open(CORPCODE_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    mapping = _download_corpcode_map()
    os.makedirs(os.path.dirname(CORPCODE_CACHE_PATH), exist_ok=True)
    with open(CORPCODE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)
    return mapping


# ---------------------------------------------------------------------------
# DART 공시 목록 조회
# ---------------------------------------------------------------------------

def list_disclosures(
    corp_code: str,
    bgn_de: str,
    end_de: str,
    page_count: int = 100,
) -> list[dict[str, Any]]:
    """
    특정 종목의 공시 목록을 기간으로 조회.

    Args:
        corp_code: DART 고유 corp_code (8자리)
        bgn_de, end_de: YYYYMMDD
    """
    url = f"{settings.DART_BASE_URL}/list.json"
    params = {
        "crtfc_key": _require_key(),
        "corp_code": corp_code,
        "bgn_de": bgn_de,
        "end_de": end_de,
        "page_count": page_count,
        "page_no": 1,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    status = data.get("status")
    # 013 = 조회된 데이터 없음 → 정상 처리
    if status == "013":
        return []
    if status != "000":
        raise RuntimeError(f"DART list API error: status={status} message={data.get('message')}")
    return data.get("list", [])
