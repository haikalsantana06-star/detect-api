"""Pydantic request/response schemas."""
from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded JPEG image")
    angle: str | None = Field(None, description="Angle name (auto-detected if omitted)")
    threshold: float | None = Field(None, ge=0.0, le=1.0)


class DeskResult(BaseModel):
    occupied: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    person: str | None = None


class DetectResponse(BaseModel):
    angle: str
    desks: dict[str, DeskResult]


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: list[str]
