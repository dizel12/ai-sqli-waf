from __future__ import annotations
import os
from pathlib import Path

from fastapi import FastAPI

from inference_svc.registry import ModelRegistry
from inference_svc.schemas import PredictRequest, PredictResponse

_DEFAULT_ART = Path(__file__).resolve().parent.parent / "ml" / "artifacts"


def create_app(artifacts_dir: Path | None = None,
               active_model: str | None = None,
               threshold: float | None = None) -> FastAPI:
    reg = ModelRegistry(
        artifacts_dir or Path(os.environ.get("ARTIFACTS_DIR", _DEFAULT_ART)),
        active_model or os.environ.get("ACTIVE_MODEL", "cnn"),
        threshold if threshold is not None
        else float(os.environ.get("BLOCK_THRESHOLD", "0.5")),
        float(os.environ.get("BEST_EFFORT_BUDGET_MS", "120")),
    )
    app = FastAPI(title="inference-svc")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok",
                "models_loaded": sorted(reg.models),
                "active_model": reg.active_model}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        return PredictResponse(results=reg.score_values(req.values))

    return app


app = create_app() if os.environ.get("INFERENCE_EAGER") else None
