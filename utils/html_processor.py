"""使用 BeautifulSoup 从 HTML 中提取标题与正文文本。"""

from __future__ import annotations

from bs4 import BeautifulSoup


def extract_title_and_body(html: str) -> tuple[str, str]:
    """
    解析 HTML，返回 (title, body_plain_text)。
    移除所有 <script> 与 <style> 后再取正文；若存在 <body> 则仅取其文本，否则取整页可见文本。
    """
    soup = BeautifulSoup(html, "html.parser")

    title_el = soup.find("title")
    title = title_el.get_text(strip=True) if title_el else ""

    for tag in soup(["script", "style"]):
        tag.decompose()

    body_el = soup.find("body")
    root = body_el if body_el else soup
    body_text = root.get_text(separator=" ", strip=True)
    # 合并多余空白，便于检索与摘要展示
    body_text = " ".join(body_text.split())

    return title, body_text
