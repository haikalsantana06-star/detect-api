"""
detect-api config module.

Loads settings from environment / .env and zones from zones.json.
Caches configuration using lru_cache.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from mysql.connector import pooling
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# DB connection pooling
# ---------------------------------------------------------------------------

_db_pool: pooling.MySQLConnectionPool | None = None


def init_db_pool(config: Config) -> None:
    """Initialize the MySQL connection pool at startup."""
    global _db_pool
    _db_pool = pooling.MySQLConnectionPool(
        pool_name="cape_pool",
        pool_size=5,
        host=config.db_host,
        port=config.db_port,
        database=config.db_name,
        user=config.db_user,
        password=config.db_password,
    )


def get_db_connection():
    """Get a connection from the pool. Caller must close when done."""
    if _db_pool is None:
        raise RuntimeError("DB pool not initialized. Call init_db_pool() first.")
    return _db_pool.get_connection()


# ---------------------------------------------------------------------------
# In-memory TTL cache for stats endpoints
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[datetime, Any]] = {}


def get_cached_stats(key: str, compute_fn, ttl_seconds: int = 60):
    """Return cached value if fresh, otherwise compute and cache it."""
    now = datetime.now()
    if key in _cache:
        ts, val = _cache[key]
        if (now - ts).total_seconds() < ttl_seconds:
            return val
    val = compute_fn()
    _cache[key] = (now, val)
    return val


def clear_stats_cache():
    """Clear the stats cache (useful after new detections are saved)."""
    global _cache
    _cache.clear()


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
    allowed_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"],
        description="Comma-separated CORS allowed origins",
    )
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
    allowed_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"])
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

    # Resolve relative paths from the app directory
    if not zones_file.is_absolute():
        zones_file = Path(__file__).parent / settings.zones_path

    # If still not found and zones_path looks like just a filename, try cwd
    if not zones_file.exists() and "/" not in str(settings.zones_path) and "\\" not in str(settings.zones_path):
        cwd_zones = Path.cwd() / settings.zones_path
        if cwd_zones.exists():
            zones_file = cwd_zones

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
        allowed_origins=settings.allowed_origins,
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
