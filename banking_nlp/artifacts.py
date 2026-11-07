"""Small persistence helpers: JSON, hashes, manifests, environment capture."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, sort_keys=False, default=_json_default)
        fh.write("\n")
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"Not JSON serialisable: {type(o)!r}")


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hash(config: dict) -> str:
    """Stable short hash of a JSON-serialisable configuration."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=_json_default)
    return sha256_text(canonical)[:12]


def row_order_checksum(row_ids: Iterable[str]) -> str:
    """Checksum of an ordered list of row IDs (used for cached embeddings/scores)."""
    return sha256_text("\n".join(row_ids))


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "packages": package_versions(
            ["numpy", "pandas", "scikit-learn", "scipy", "spacy", "en_core_web_sm",
             "sentence-transformers", "transformers", "torch", "joblib", "matplotlib"]
        ),
    }
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except Exception:  # not a git checkout, or git missing
        info["git_commit"] = None
    return info
