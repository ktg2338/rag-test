from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    RecursiveCharacterTextSplitter를 사용하여 문장/문단 경계를 존중하며 청킹.
    분할 우선순위: 문단 -> 문장 -> 단어 -> 문자
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",  # 문단
            "\n",    # 줄바꿈
            "。",    # 한국어/일본어 마침표
            ".",     # 영어 마침표
            "!",
            "?",
            ";",
            ",",
            " ",     # 공백
            "",      # 문자
        ],
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_text(text)
