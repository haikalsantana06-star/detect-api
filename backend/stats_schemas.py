"""Pydantic schemas for stats endpoints and 5 management indicators."""
from pydantic import BaseModel, Field
from typing import Any


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
