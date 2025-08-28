from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


# ─── Send all loguru output to stderr ─────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}",
    colorize=True,
)


def _stdin_has_data() -> bool:
    """
    Robustly detect whether stdin has data.
    """
    try:
        return not sys.stdin.isatty()
    except Exception:
        return False


def read_df(path: Optional[Path]) -> pd.DataFrame:
    """
    Read CSV from a path if provided, else from stdin when data is present.
    """
    if path is not None:
        logger.info(f"Reading DataFrame from {path}...")
        return pd.read_csv(path)

    if _stdin_has_data():
        logger.info("Reading DataFrame from stdin...")
        data = sys.stdin.buffer.read()
        return pd.read_csv(io.BytesIO(data))
    raise ValueError("No input provided. Pass a CSV path or pipe data into stdin.")


def write_df(df: pd.DataFrame) -> None:
    """
    Always write to stout. Users can redirect with '>' to a file or pipe onward.
    """
    csv_bytes = df.to_csv(index=False).encode()
    sys.stdout.buffer.write(csv_bytes)
    sys.stdout.flush()
    logger.success("CSV written to stdout.")


def comma_split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def comma_split_int(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]
