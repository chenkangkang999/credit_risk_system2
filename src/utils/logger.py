import os
import re
import sys
from pathlib import Path

from loguru import logger

_configured = False


def _hide_secrets(record):
    message = record["message"]
    secret = os.getenv("DEEPSEEK_API_KEY", "")
    if secret:
        message = message.replace(secret, "[已隐藏]")
    record["message"] = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[已隐藏]", message)


def get_logger(name="credit_risk", log_dir="./logs", level="INFO", rotation="10 MB", retention="14 days"):
    global _configured
    if not _configured:
        folder = Path(log_dir)
        if not folder.is_absolute():
            folder = Path(__file__).resolve().parents[2] / folder
        folder.mkdir(parents=True, exist_ok=True)
        logger.remove()
        logger.configure(patcher=_hide_secrets)
        fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[module]} | {message}"
        logger.add(sys.stderr, format=fmt, level=os.getenv("LOG_LEVEL", level))
        logger.add(folder / "system_{time:YYYY-MM-DD}.log", format=fmt, level=level,
                   rotation=rotation, retention=retention, encoding="utf-8", diagnose=False)
        _configured = True
    return logger.bind(module=name)


default_logger = get_logger()
