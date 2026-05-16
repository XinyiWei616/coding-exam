"""统一加载 .env；全部使用 override=True，避免 shell 空变量占位导致读不到 GEMINI_API_KEY。"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# services/ 的上一级为项目根（与 main.py、.env 同级）
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_application_dotenv() -> None:
    """
    依次加载项目根 .env、.env.local，再从 cwd 向上查找 .env；均使用 override=True。
    """
    root_env = PROJECT_ROOT / ".env"
    if root_env.is_file():
        load_dotenv(root_env, override=True)
    else:
        logger.warning("未找到项目根 .env: %s", root_env)

    local = PROJECT_ROOT / ".env.local"
    if local.is_file():
        load_dotenv(local, override=True)

    load_dotenv(override=True)


def normalize_api_key(raw: str | None) -> str:
    if raw is None:
        return ""
    return raw.strip().lstrip("\ufeff").strip('"\'')


def gemini_api_key_from_env() -> str:
    return normalize_api_key(
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )
