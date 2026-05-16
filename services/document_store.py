"""内存文档表：与 Chroma 向量库在业务上保持一致（启动时由 data/ 引导）。"""

from __future__ import annotations

import re
from typing import Iterator

_documents: dict[str, dict[str, str]] = {}

# 与 POST /v1/documents 的自定义 id 规则一致，便于 data/*.html 文件名作为 id
DOCUMENT_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,255}$")


def clear_store() -> None:
    _documents.clear()


def put_document(doc_id: str, title: str, body: str) -> None:
    _documents[doc_id] = {"title": title, "body": body}


def get_document(doc_id: str) -> dict[str, str] | None:
    return _documents.get(doc_id)


def iter_documents() -> Iterator[tuple[str, dict[str, str]]]:
    return iter(_documents.items())
