from __future__ import annotations
import json, sys
import os
from pathlib import Path
from functools import lru_cache

import numpy as np

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import BACKGROUND, FEATURES, META, MODEL, POSITIVE_CLASS, positive_class_probability
from backend.schemas import PatientInput, PredictionResponse

app = FastAPI(title='Cardia Heart Risk API', version='1.0.0', description='Educational ML decision-support prototype.')
default_origins = 'http://localhost:5173,http://127.0.0.1:5173'
allowed_origins = [
    origin.strip().rstrip('/')
    for origin in os.getenv('CARDIA_CORS_ORIGINS', default_origins).split(',')
    if origin.strip()
]
if '*' in allowed_origins:
    raise RuntimeError('CARDIA_CORS_ORIGINS must list explicit trusted origins; wildcard CORS is not allowed.')
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False, allow_methods=['GET', 'POST'], allow_headers=['*'],
)

@lru_cache(maxsize=1)
def resources():
    if not MODEL.exists() or not META.exists():
        raise RuntimeError('Model artifacts are missing. Run the training pipeline first.')
    meta = json.loads(META.read_text())
    trained_version = meta.get('versions', {}).get('scikit-learn')
    if trained_version and trained_version != sklearn.__version__:
        raise RuntimeError(
            f"scikit-learn version mismatch: model expects {trained_version}, "
            f"but this environment has {sklearn.__version__}. Install the pinned backend requirements."
        )
    return joblib.load(MODEL), meta

@app.get('/api/health')
def health():
    try:
        _, meta = resources()
        return {'status': 'ok', 'model_loaded': True, 'model_name': meta.get('selected_model', 'Unknown')}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.get('/api/model')
def model_info():
    try:
        _, meta = resources()
        return {
            'selected_model': meta.get('selected_model'),
            'n_train': meta.get('n_train'), 'n_test': meta.get('n_test'),
            'test_metrics': meta.get('test_metrics', {}),
            'cv_table': meta.get('cv_table', []),
            'versions': meta.get('versions', {}),
            'dataset': 'Cleveland heart disease recoded dataset (303 records before duplicate removal)',
            'features': FEATURES,
            'warning': 'Small, single-source historical dataset. Not clinically validated.'
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

@app.get('/api/input-schema')
def input_schema():
    """Expose the exact accepted fields and constraints from the request model."""
    properties = PatientInput.model_json_schema()['properties']
    return {feature: properties[feature] for feature in FEATURES}


@app.post('/api/explain')
def explain(payload: PatientInput):
    """Return additive local explanations in model log-odds space.

    SHAP values are grouped back to the 13 original clinical inputs. They describe
    this model's associations, not causal or clinical effects.
    """
    try:
        import shap
        pipe, meta = resources()
        if meta.get('selected_model') != 'Logistic Regression':
            raise HTTPException(status_code=501, detail='Local linear SHAP is currently configured for the Logistic Regression artifact only.')
        if not BACKGROUND.exists():
            raise HTTPException(status_code=503, detail='SHAP background data is missing.')
        background = pd.read_csv(BACKGROUND)[FEATURES]
        x = pd.DataFrame([payload.model_dump()], columns=FEATURES)
        prep = pipe.named_steps['prep']
        estimator = pipe.named_steps['model']
        background_transformed = prep.transform(background)
        patient_transformed = prep.transform(x)
        explainer = shap.LinearExplainer(estimator, background_transformed)
        explanation = explainer(patient_transformed)
        values = np.asarray(explanation.values)
        class_index = list(estimator.classes_).index(POSITIVE_CLASS)
        if values.ndim == 3:
            values = values[0, :, class_index]
        elif values.ndim == 2:
            values = values[0]
        transformed_names = prep.get_feature_names_out()
        grouped = {feature: 0.0 for feature in FEATURES}
        for name, value in zip(transformed_names, values):
            source = str(name).split('__', 1)[-1]
            for feature in FEATURES:
                if source == feature or source.startswith(feature + '_'):
                    grouped[feature] += float(value)
                    break
        base = np.asarray(explanation.base_values)
        if base.ndim >= 2:
            base_value = float(base.reshape(-1, base.shape[-1])[0, class_index])
        else:
            base_value = float(base.reshape(-1)[0])
        expected_log_odds = float(estimator.decision_function(patient_transformed)[0])
        if not np.isclose(base_value + sum(grouped.values()), expected_log_odds, atol=1e-5):
            raise RuntimeError('SHAP contributions do not add up to the positive-class model output.')
        return {
            'output_space': 'log_odds',
            'base_value': base_value,
            'contributions': [{'feature': key, 'value': value} for key, value in grouped.items()],
            'note': 'SHAP contributions explain the positive heart-disease dataset class in log-odds space. Positive values move the model output toward that class; negative values move it away. They are model associations, not medical causes.'
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Explanation failed: {exc}')

@app.post('/api/predict', response_model=PredictionResponse)
def predict(payload: PatientInput):
    try:
        pipe, meta = resources()
        row = pd.DataFrame([payload.model_dump()], columns=FEATURES)
        probability = float(positive_class_probability(pipe, row)[0])
        return PredictionResponse(
            probability=probability,
            model_name=meta.get('selected_model', 'Unknown'),
            model_version='heart-model-v1',
            disclaimer='Educational estimate only. This output is not a diagnosis, screening result, or treatment recommendation.'
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {exc}')
