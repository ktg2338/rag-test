"""
국내 주식 공시(DART) ETL 진입점.

흐름:
  1) pykrx로 시총 상위 N개 종목 선정 (KOSPI / KOSDAQ)
  2) DART corpCode 매핑으로 stock_code → corp_code 변환
  3) 각 종목의 최근 N일 공시 목록 조회
  4) 이미 처리한 rcept_no는 스킵 (sqlite seen store)
  5) 신규 공시의 본문(document.xml) 다운로드 및 텍스트 추출
  6) 청킹 후 ChromaDB upsert + BM25 인덱스 재빌드
  7) 처리 완료된 rcept_no를 seen store에 기록

사용 예:
    python -m scripts.ingest_stocks --top 10 --days 7
    python -m scripts.ingest_stocks --top 50 --days 30 --market KOSDAQ
"""

import argparse
import time
from datetime import datetime, timedelta

from app.services.bm25_index import bm25_index
from app.services.stocks.extractor import fetch_document_text
from app.services.stocks.sources import (
    get_corpcode_map,
    get_top_stocks,
    list_disclosures,
)
from app.services.stocks.state import SeenStore
from app.services.stocks.transformer import build_records
from app.services.vectorstore import get_all_documents, upsert_texts


def run(
    top_n: int = 10,
    days: int = 7,
    market: str = "KOSPI",
    sleep_sec: float = 0.2,
    rebuild_bm25: bool = True,
) -> None:
    end_dt = datetime.now()
    bgn_dt = end_dt - timedelta(days=days)
    end_de = end_dt.strftime("%Y%m%d")
    bgn_de = bgn_dt.strftime("%Y%m%d")

    print(f"[1/5] 시총 상위 {top_n}개 ({market}) 종목 조회")
    stocks = get_top_stocks(top_n=top_n, market=market)
    print(f"      → {len(stocks)}개 종목")

    print("[2/5] DART corpCode 매핑 로드")
    corp_map = get_corpcode_map()
    print(f"      → {len(corp_map)}개 매핑")

    seen = SeenStore()

    print(f"[3/5] 공시 목록 조회 ({bgn_de} ~ {end_de})")
    pending: list[tuple[dict, dict]] = []  # (disclosure, stock)
    for s in stocks:
        meta = corp_map.get(s["ticker"])
        if not meta:
            print(f"      ! corp_code 미발견: {s['ticker']} {s['name']}")
            continue
        try:
            items = list_disclosures(meta["corp_code"], bgn_de, end_de)
        except Exception as e:
            print(f"      ! list 조회 실패 {s['name']}: {e}")
            continue

        new_items = [d for d in items if not seen.has(d["rcept_no"])]
        print(f"      - {s['name']}({s['ticker']}): 전체 {len(items)}건, 신규 {len(new_items)}건")
        for d in new_items:
            pending.append((d, s))
        time.sleep(sleep_sec)  # rate limit

    if not pending:
        print("[4/5] 신규 공시 없음. 종료.")
        return

    print(f"[4/5] 신규 공시 {len(pending)}건 본문 다운로드 및 인덱싱")
    all_texts: list[str] = []
    all_metas: list[dict] = []
    processed: list[tuple[str, str]] = []
    failed_body = 0

    for i, (disclosure, stock) in enumerate(pending, 1):
        rcept_no = disclosure["rcept_no"]
        body = fetch_document_text(rcept_no)
        if body is None:
            failed_body += 1
        texts, metas = build_records(disclosure, body, stock)
        all_texts.extend(texts)
        all_metas.extend(metas)
        processed.append((rcept_no, disclosure.get("corp_code", "")))
        if i % 10 == 0 or i == len(pending):
            print(f"      ... {i}/{len(pending)} (본문없음 누적 {failed_body})")
        time.sleep(sleep_sec)

    print(f"      → 총 청크 {len(all_texts)}개, 본문 추출 실패 {failed_body}건 (제목/메타로 인덱싱됨)")

    print("[5/5] ChromaDB upsert + BM25 재빌드")
    ids = upsert_texts(all_texts, metadatas=all_metas)
    seen.mark_many(processed)
    print(f"      → upsert {len(ids)}개")

    if rebuild_bm25:
        all_docs = get_all_documents().get("documents", [])
        if all_docs:
            bm25_index.build(all_docs)
            print(f"      → BM25 재빌드 ({bm25_index.doc_count} docs)")

    print("완료.")


def main() -> None:
    parser = argparse.ArgumentParser(description="국내 주식 DART 공시 ETL")
    parser.add_argument("--top", type=int, default=10, help="시총 상위 N개 (기본 10)")
    parser.add_argument("--days", type=int, default=7, help="최근 N일 공시 (기본 7)")
    parser.add_argument(
        "--market",
        choices=["KOSPI", "KOSDAQ"],
        default="KOSPI",
        help="시장 구분 (기본 KOSPI)",
    )
    parser.add_argument(
        "--no-bm25",
        action="store_true",
        help="BM25 재빌드 생략 (대량 적재 시 마지막 1회만 권장)",
    )
    args = parser.parse_args()

    run(
        top_n=args.top,
        days=args.days,
        market=args.market,
        rebuild_bm25=not args.no_bm25,
    )


if __name__ == "__main__":
    main()
