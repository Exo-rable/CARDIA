from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

class PatientInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    age: int = Field(ge=20, le=90)
    sex: Literal[0, 1]
    cp: Literal[0, 1, 2, 3]
    trestbps: int = Field(ge=80, le=220)
    chol: int = Field(ge=100, le=600)
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: int = Field(ge=60, le=220)
    exang: Literal[0, 1]
    oldpeak: float = Field(ge=0, le=7)
    slope: Literal[0, 1, 2]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[1, 2, 3]

class PredictionResponse(BaseModel):
    probability: float
    model_name: str
    model_version: str
    disclaimer: str
