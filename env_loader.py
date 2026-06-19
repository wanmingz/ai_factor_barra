# env_loader.py — load key=value pairs from a local .env file into os.environ

import os
from pathlib import Path

from config import ENV_FILE


def load_env_file(path: str | Path | None = None) -> None:
    """Load variables from .env without overwriting existing os.environ entries."""
    env_path = Path(path or ENV_FILE)
    if not env_path.is_file():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
