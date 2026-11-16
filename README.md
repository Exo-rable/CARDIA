# Cardia — Heart Risk Prediction Project

A React + FastAPI application built around the supplied Cleveland heart-disease ML pipeline. The original Streamlit app remains in `app.py` as a legacy reference; the new product UI is in `frontend/` and the inference API is in `backend/`.

> **Important:** This is an educational research prototype, not a medical device, diagnosis, screening tool, or treatment recommendation. Do not enter identifiable patient data or use predictions for clinical decisions.

## Project layout

```text
heart-risk/
├── backend/                 # FastAPI inference API + Pydantic request schema
├── frontend/                # React, Vite, TypeScript, Framer Motion, Recharts
├── data/raw/                # supplied source dataset
├── data/clean/              # cleaned dataset
├── models/                  # saved sklearn pipeline + metadata + SHAP background
├── reports/                 # evaluation tables and figures
├── src/                     # original cleaning, EDA, training and explainability
├── tests/                   # existing project smoke test
├── app.py                   # original Streamlit interface (legacy)
├── run_backend.sh
└── run_frontend.sh
```

## Step-by-step local setup

### 1. Create a Python environment

Use the Python version used by the original project (Python 3.13) and create a virtual environment:

**Windows PowerShell**
```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

**macOS / Linux**
```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

The saved artifact metadata records scikit-learn `1.9.1`. The API checks this version before loading the joblib model; do not ignore version warnings or load a pickle/joblib model from an untrusted source.

### 2. Start the backend

From the project root, with the virtual environment activated:
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/api/health` and interactive API docs at `http://127.0.0.1:8000/docs`.

### 3. Start the React frontend

Open a second terminal:
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The default API URL is `http://localhost:8000`. If needed, create `frontend/.env.local` containing:
```env
VITE_API_URL=http://localhost:8000
```
Restart Vite after changing environment variables.

### 4. Deploy with Docker Compose

Install Docker Engine/Desktop with the Compose plugin, then from the repository root run:
```bash
docker compose up --build -d
```

Open `http://localhost:8080`. Nginx serves the built React app and forwards `/api` requests to the private FastAPI service. The frontend binds to `127.0.0.1` by default, so it is reachable only from the host. Set `CARDIA_HTTP_PORT` to change the local host port. To stop the services, run `docker compose down`.

The Compose stack is a deployable container baseline, not a complete public-production security boundary. For a remote deployment, set `CARDIA_HTTP_BIND_ADDRESS=0.0.0.0` only when it is behind a trusted HTTPS ingress/reverse proxy. CARDIA is an unauthenticated educational prototype and must only receive synthetic demonstration data; do not use it to store or process real patient information. See [Before any real-world patient use](#before-any-real-world-patient-use) before considering a clinical or patient-facing deployment.

For separate frontend/API hosting, build with `VITE_API_URL` set to the HTTPS API origin, and set `CARDIA_CORS_ORIGINS` on the backend to the exact HTTPS frontend origin. The Vite build embeds this value; rebuild after changing it. Wildcard CORS is rejected. When the frontend and API share an origin through a reverse proxy, leave `VITE_API_URL` empty.

## API endpoints

- `GET /api/health` — checks that the saved model can load.
- `GET /api/model` — returns model metadata for the optional advanced-information panel.
- `GET /api/input-schema` — exposes the accepted fields, numeric bounds, and category options used by the assessment form.
- `POST /api/predict` — validates the 13 feature values and returns the positive-class model probability.
- `POST /api/explain` — returns local SHAP contributions for the submitted information.

The frontend never reimplements preprocessing or model inference. The API loads `models/heart_model.joblib`, builds a row in the exact feature order, and selects the probability column by the estimator's positive class label (`1`), not by an assumed column position. The result view uses the project's 0.50 model decision cut-off; this is not a clinical threshold.

The API allows the two local Vite origins by default. For deployment, set `CARDIA_CORS_ORIGINS` to a comma-separated list of exact trusted frontend origins (for example, `https://cardia.example.org`). Wildcard CORS is rejected.

## Data coding note

The project uses a recoded copy of the Cleveland dataset. The feature codes remain unchanged. Their category labels follow the source dataset documentation: chest-pain code `0` is typical angina, `1` atypical angina, `2` non-anginal pain, and `3` asymptomatic. Resting ECG codes are `0` normal, `1` ST-T wave abnormality, and `2` probable or definite left-ventricular hypertrophy. Exercise-test slope codes are `0` upsloping, `1` flat, and `2` downsloping. Fasting blood sugar is coded as `1` only when the source value is above 120 mg/dL; `0` means 120 mg/dL or lower. Do not interpret these dataset categories as diagnoses; use the original report and a healthcare professional for clinical interpretation.

## Model evaluation caveat

The included metadata reports a 61-record project test split. The original training script's prior version also evaluated alternative candidates on this split to generate a comparison CSV, so the saved metrics are not an untouched independent validation. Treat them as exploratory and re-evaluate using a fresh external test cohort before making stronger claims. For this saved evaluation, the model correctly labelled 51 of 61 records (83.6% accuracy); it identified 22 of 28 class-1 records (78.6% sensitivity), correctly labelled 29 of 33 class-0 records (87.9% specificity), and 22 of 26 model class-1 labels matched class 1 in the dataset (84.6% precision). These figures and their counts are kept in the optional advanced-information section to make the small sample and error types visible without putting model-performance metrics on the home page. For a future rigorous study, perform all model and threshold selection using training-only nested cross-validation, freeze the model, then evaluate once on an independent external dataset.

## Current UI scope

Included: responsive patient-first dashboard, sidebar navigation, guided four-step assessment using backend-provided input constraints, API-backed prediction and SHAP explanation, browser-local demonstration history, measurement chart, plain-language learning centre, optional advanced model information, loading/error states, and reduced-motion support.

Included: local SHAP explanations for the saved Logistic Regression pipeline, grouped back to the 13 original inputs and explicitly labelled as log-odds contributions. Next engineering increments: PDF report export, authentication/authorization if records are ever stored, audit logging, external validation, subgroup evaluation, and deployment security review. Do not represent feature associations as causes or the raw model probability as clinically calibrated risk.

The patient-facing result shows both model outputs for the two historical dataset patterns, rather than only the winning output. These are model outputs, not calibrated personal probabilities or medical risk estimates. The result presents six influential measurement explanations and eight directional contribution bars based on local SHAP values; a collapsed section contains explanations for all 13 submitted answers, and a separate advanced section contains numerical SHAP details. These describe model behavior, not medical causes. No PDF report generator is present in this React application; report export remains future work.

## Patient-first UI update (CARDIA 1.1)

The overview includes an optional display name, actual browser-demo assessment counts and dates, a chart based on saved submitted readings, history actions, and a plain-language learning centre. The assessment is a four-step flow; it validates against the live backend input schema and uses the existing prediction and SHAP endpoints. Field-specific explanations and accepted input limits are available on demand in an accessible drawer, keeping the form itself focused on entering answers. Model performance details are kept inside an optional advanced-information section.

### Local history and privacy boundary

Assessment history and the optional display name are stored in the current browser's `localStorage`. This is a convenience for a local academic demo only: it is not encrypted health-record storage, not an authenticated patient account, not backed up, and not shared across devices. Use synthetic demonstration values only. A real patient-facing release must not use this local-storage feature as a medical record system.

### Run the upgraded app (Windows PowerShell)

Terminal 1, from the project root:

```powershell
.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The backend must use the pinned versions in `backend/requirements.txt`; the saved joblib artifact records scikit-learn `1.9.1` and the API intentionally refuses a version mismatch.

### Before any real-world patient use

This release is still an academic research prototype. It is not suitable for clinical monitoring, screening, diagnosis, triage or treatment decisions. A genuine patient-facing deployment requires, at minimum: prospective and external validation on representative populations; calibration and threshold validation; subgroup/fairness and missing-data evaluation; clinical governance and clinician review; threat modelling and security testing; authenticated access, role-based authorization, encryption in transit and at rest, consent and retention controls, audit trails, incident response, accessibility testing, and applicable legal/regulatory review. The current local demo history must be replaced by a secure, governed data architecture before handling patient information.
