from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import EMBARKED_VALUES, PCLASS_VALUES, SEX_VALUES
from src.predict import load_model, predict_passenger


class PassengerInput(BaseModel):
    """Données d'un passager. L'âge est la seule donnée facultative."""

    model_config = ConfigDict(extra="forbid")

    pclass: int
    sex: str
    age: float | None = None
    sibsp: int = Field(ge=0)
    parch: int = Field(ge=0)
    fare: float = Field(ge=0)
    embarked: str

    @field_validator("pclass")
    @classmethod
    def validate_pclass(cls, value: int) -> int:
        if value not in PCLASS_VALUES:
            allowed = ", ".join(map(str, sorted(PCLASS_VALUES)))
            raise ValueError(f"pclass invalide : choisissez parmi {allowed}")
        return value

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, value: str) -> str:
        if value not in SEX_VALUES:
            allowed = ", ".join(map(str, sorted(SEX_VALUES)))
            raise ValueError(f"sex invalide : choisissez parmi {allowed}")
        return value

    @field_validator("embarked")
    @classmethod
    def validate_embarked(cls, value: str) -> str:
        if value not in EMBARKED_VALUES:
            allowed = ", ".join(map(str, sorted(EMBARKED_VALUES)))
            raise ValueError(f"embarked invalide : choisissez parmi {allowed}")
        return value


class PredictionOutput(BaseModel):
    survived: bool
    probability: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Le modèle est chargé une seule fois au démarrage du processus.
    app.state.model = load_model()
    yield
    app.state.model = None


app = FastAPI(
    title="Titanic Passenger API",
    description="API de prédiction de survie des passagers du Titanic.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {"name": "Titanic Passenger API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé.")
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PassengerInput, request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Le modèle n'est pas chargé.")

    passenger = payload.model_dump()
    try:
        # Contrat attendu : predict_passenger(model, passenger_dict).
        # La logique de prédiction reste exclusivement dans src/predict.py.
        result = predict_passenger(model, passenger)
        return PredictionOutput(**result)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Impossible de produire la prédiction : {exc}",
        ) from exc
