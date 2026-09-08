from __future__ import annotations
from pydantic import BaseModel


class PredictRequest(BaseModel):
    values: list[str]


class PerValue(BaseModel):
    value: str
    scores: dict[str, float | None]
    active_model: str
    score: float
    decision: str
    threshold: float


class PredictResponse(BaseModel):
    results: list[PerValue]
