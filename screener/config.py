"""Configuration loading and small shared helpers."""
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone

import yaml

# Repo root = parent of the `screener` package directory.
ROOT = Path(__file__).resolve().parents[1]


class Config:
    """Thin attribute-style wrapper around config.yml with path resolution."""

    def __init__(self, data: dict, root: Path = ROOT):
        self._d = data
        self.root = root

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Config":
        path = Path(path) if path else ROOT / "config.yml"
        with open(path, "r", encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    def __getitem__(self, key):
        return self._d[key]

    def get(self, key, default=None):
        return self._d.get(key, default)

    # -- convenience blocks --------------------------------------------------
    @property
    def strategy(self):
        return self._d["strategy"]

    @property
    def filters(self):
        return self._d["filters"]

    @property
    def learning(self):
        return self._d["learning"]

    @property
    def data(self):
        return self._d["data"]

    @property
    def weights(self) -> dict:
        return dict(self._d["factor_weights"])

    def path(self, key: str) -> Path:
        """Resolve a configured relative path against the repo root."""
        return self.root / self._d["paths"][key]

    def save_weights(self, weights: dict) -> None:
        """Persist updated factor weights back into config.yml in place."""
        cfg_path = self.root / "config.yml"
        with open(cfg_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        raw["factor_weights"] = {k: round(float(v), 4) for k, v in weights.items()}
        with open(cfg_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(raw, fh, sort_keys=False, allow_unicode=True)
        self._d["factor_weights"] = raw["factor_weights"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def today_iso() -> str:
    return utcnow().strftime("%Y-%m-%d")


def ensure_dirs(cfg: Config) -> None:
    for key in ("state_dir", "cache_dir"):
        cfg.path(key).mkdir(parents=True, exist_ok=True)
