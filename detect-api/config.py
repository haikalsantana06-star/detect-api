"""
detect-api config module.

Loads settings from environment / .env and zones from zones.json.
Caches configuration using lru_cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# DB settings
# ---------------------------------------------------------------------------


class DBSettings(BaseSettings):
    """Database connection settings for MySQL."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "localhost"
    port: int = 3306
    name: str = "upitas_mon"
    user: str = ""
    password: str = ""


# ---------------------------------------------------------------------------
# Top-level settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    threshold: float = 0.5
    device: str = "cpu"
    model_dir: str = "models"
    zones_path: str = "zones.json"
    db: DBSettings = Field(default_factory=DBSettings)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Zone:
    """ROI zone defining a desk region in an image."""

    x: int
    y: int
    w: int
    h: int


@dataclass
class Config:
    """
    Full application configuration assembled from Settings and zones.json.
    """

    # zones[angle][desk_name] -> Zone
    zones: dict[str, dict[str, Zone]] = field(default_factory=dict)
    desks: list[str] = field(default_factory=list)
    person_map: dict[str, str] = field(default_factory=dict)
    threshold: float = 0.5
    device: str = "cpu"
    model_dir: str = "models"
    db_host: str = "localhost"
    db_port: int = 3306
    db_name: str = "upitas_mon"
    db_user: str = ""
    db_password: str = ""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(settings: Settings) -> Config:
    """
    Load zones.json and build a Config instance.

    Parameters
    ----------
    settings : Settings
        Already-loaded application settings (provides zones_path etc.).

    Returns
    -------
    Config
        Fully-populated configuration object.
    """
    zones_file = Path(settings.zones_path)

    # Resolve relative paths from the detect-api directory
    if not zones_file.is_absolute():
        zones_file = Path(__file__).parent.parent / settings.zones_path

    with open(zones_file, encoding="utf-8") as fh:
        raw: dict[str, Any] = json.load(fh)

    # Parse zones into Zone dataclasses: zones[angle][desk] = Zone(...)
    parsed_zones: dict[str, dict[str, Zone]] = {}
    for angle, desks in raw.get("zones", {}).items():
        parsed_zones[angle] = {}
        for desk_name, rect in desks.items():
            parsed_zones[angle][desk_name] = Zone(
                x=rect["x"],
                y=rect["y"],
                w=rect["w"],
                h=rect["h"],
            )

    db = settings.db

    return Config(
        zones=parsed_zones,
        desks=raw.get("desks", []),
        person_map=raw.get("person_map", {}),
        threshold=settings.threshold,
        device=settings.device,
        model_dir=settings.model_dir,
        db_host=db.host,
        db_port=db.port,
        db_name=db.name,
        db_user=db.user,
        db_password=db.password,
    )


# ---------------------------------------------------------------------------
# Cached accessors
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return a cached Config singleton."""
    return load_config(get_settings())


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    settings = get_settings()
    print(f"[settings]  threshold={settings.threshold}  device={settings.device}")
    print(f"[settings]  model_dir={settings.model_dir}  zones_path={settings.zones_path}")
    print(f"[settings]  db.host={settings.db.host}  db.port={settings.db.port}  db.name={settings.db.name}")

    cfg = get_config()
    print(f"[config]    desks={cfg.desks}")
    print(f"[config]    person_map={cfg.person_map}")
    print(f"[config]    angles={list(cfg.zones.keys())}")
    for angle, desks in cfg.zones.items():
        for desk, zone in desks.items():
            print(f"[config]      {angle}/{desk}: Zone(x={zone.x}, y={zone.y}, w={zone.w}, h={zone.h})")
    print("[OK] config loaded successfully")
