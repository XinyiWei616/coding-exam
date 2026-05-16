"""将纯文本按段落与最大长度切分为检索块。"""

from __future__ import annotations

import re


def chunk_plain_text(text: str, max_chars: int = 300) -> list[str]:
    """
    先按空行分段，再对超长段落按 max_chars 硬切。
    段内空白压成单空格，便于与 HTML 提取结果配合。
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    parts = re.split(r"\n\s*\n+", text)
    paragraphs = [p.strip() for p in parts if p.strip()]
    if not paragraphs:
        paragraphs = [re.sub(r"\s+", " ", text).strip()]

    chunks: list[str] = []
    for p in paragraphs:
        flat = re.sub(r"\s+", " ", p).strip()
        if len(flat) <= max_chars:
            chunks.append(flat)
            continue
        for i in range(0, len(flat), max_chars):
            piece = flat[i : i + max_chars].strip()
            if piece:
                chunks.append(piece)
    return chunks
