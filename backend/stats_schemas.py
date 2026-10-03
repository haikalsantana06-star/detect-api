"""Pydantic schemas for stats endpoints and 5 management indicators."""
from enum import Enum
from typing import Any

from pydantic import BaseModel


class Indicator(BaseModel):
    label: str
    value: Any  # bool | str | int | float
    unit: str | None = None
    description: str


class WorkHours(BaseModel):
    start: str = "07:00"
    end: str = "17:00"
    total_minutes: int = 600


class StatsResponse(BaseModel):
    person: str
    date: str
    work_hours: WorkHours
    indicators: dict[str, Indicator]


class DeskStatsResponse(BaseModel):
    desk: str
    date: str
    work_hours: WorkHours
    indicators: dict[str, Indicator]


# ---------------------------------------------------------------------------
# Enhanced Stats Schemas
# ---------------------------------------------------------------------------


class PresenceStatus(str, Enum):
    NO_DATA = "no_data"       # 0 rows in detection_logs for this scope
    ABSENT = "absent"         # rows exist, person never detected as occupied
    PRESENT = "present"        # >=1 occupied detection during work hours


class DailyStats(BaseModel):
    """Per-person-per-day aggregation — the foundation for all rollups."""
    person: str
    date: str                  # YYYY-MM-DD in Asia/Jakarta
    has_data: bool
    presence_status: PresenceStatus
    first_seen: str | None     # HH:MM or None
    last_seen: str | None      # HH:MM or None
    work_minutes: int          # min((last_seen - first_seen), work_window_minutes)
    productive_minutes: int     # occupied frames during 07:00-17:00 × interval
    idle_minutes: int          # occupied frames outside 07:00-17:00 × interval


class AggregateStats(BaseModel):
    """Weekly/Monthly rollup."""
    scope: str                 # "weekly" | "monthly"
    start_date: str
    end_date: str
    person: str | None         # None = all people combined
    total_days: int            # days with data
    days_present: int          # days where presence_status = PRESENT
    attendance_rate: float     # days_present / total_days (0.0-1.0)
    avg_work_minutes: float   # average over days with data
    avg_productive_minutes: float
    avg_idle_minutes: float
    on_time_count: int         # days arriving <= work_start + tolerance
    on_time_rate: float        # on_time_count / days_present
    avg_arrival_time: str      # HH:MM average over present days


class TrendPoint(BaseModel):
    """One day in a trend series."""
    date: str
    person: str
    presence_status: PresenceStatus
    work_minutes: int
    productive_minutes: int
    arrival_time: str | None


class AvailableDates(BaseModel):
    """Dates that have detection data."""
    dates: list[str]           # YYYY-MM-DD, sorted desc (latest first)


class PersonComparison(BaseModel):
    """Per-person summary for 'Semua orang' view."""
    person: str
    attendance_rate: float
    avg_work_minutes: float
    avg_arrival_time: str
    on_time_rate: float
