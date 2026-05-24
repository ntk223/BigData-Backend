# PredictCare AI - CDSS Dashboard (Team 10)
## Clinical Decision Support System (CDSS) Backend API

This backend provides a clean, production-grade REST API supporting two primary clinical prediction tasks:
1. **30-day Readmission Risk Prediction** (using an L1/L2 Regularized XGBoost Classifier).
2. **12-month Mortality Risk Prediction** (using an XGBoost Survival Embeddings / XGBSE model).

Both tasks share the same 236 input patient features, but utilize different predictive algorithms, imputation schemes, and survival curve computations.

---

### 📂 Directory Structure

```text
bich-da-ta/
├── api/
│   ├── __init__.py          # Marks the folder as a Python package
│   ├── main.py              # FastAPI initialization, routing, and endpoints
│   ├── schemas.py           # Pydantic schemas (readmission and mortality response schemas)
│   └── services.py          # Preprocessing, model inference, and What-If analysis for both tasks
├── models/
│   ├── metadata.json        # Mappings, evaluation metrics, and feature names order
│   ├── readmission_model.pkl# Serialized XGBClassifier object for readmission
│   └── mortality_models_export/
│       └── mortality_models/
│           ├── final_xgbse_model.joblib   # Serialized XGBSEStackedWeibull model
│           └── fitted_imputer.joblib      # Pre-fitted SimpleImputer for mortality features
├── app.py                   # Root-level entrypoint wrapper importing api.main:app
├── requirements.txt         # Core dependencies (including xgbse and lifelines)
├── lenh.txt                 # Script commands for easy reference
└── README.md                # Documentation and API guide (this file)
```

---

### 🚀 Getting Started

#### 1. Activate the Virtual Environment

Create the virtual environment:
```bash
python -m venv .venv
```
or
```bash
python3 -m venv .venv
```
Activate the pre-configured Python virtual environment:
```bash
source .venv/bin/activate
```

#### 2. Start the Backend API Server
Run the Uvicorn development server:
```bash
uvicorn app:app --reload
```
The server will start on `http://127.0.0.1:8000`.

---

### 🔌 API Endpoints Reference

View the interactive Swagger UI documentation at:
👉 **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

#### 1. Common Endpoints
* **`GET /`**: Health check and metadata tasks description.
* **`GET /metadata`**: Retrieve training metadata, features order, and mappings.
* **`GET /sample`**: Generate a pre-filled JSON body with all 236 features. **Use this to avoid manually typing out 236 parameters when testing.**

#### 2. 30-day Readmission Endpoints
* **`POST /predict/readmission`**: Predicts 30-day readmission probability and returns a risk accumulation curve.
* **`POST /what-if/readmission`**: Simulates the 30-day readmission risk across three discharge scenarios (HOME, HOME HEALTH CARE, SKILLED NURSING FACILITY).

#### 3. 12-month Mortality Endpoints
* **`POST /predict/mortality`**: Predicts 12-month mortality risk (`mortality_risk_12m`), Restricted Mean Survival Time (`rmst_12m` in days), and returns a survival probability curve over a 12-month timeline.
* **`POST /what-if/mortality`**: Simulates the 12-month mortality curve, mortality risk, and RMST across the three discharge scenarios.

---

### 🧪 Testing with curl

Run the following commands in your terminal to test the API endpoints:

#### Set Up Sample Request Payload
```bash
curl -s http://127.0.0.1:8000/sample > sample.json
```

#### Test Readmission Endpoints
```bash
# Predict Readmission
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @sample.json

# Readmission What-If
curl -X 'POST' \
  'http://127.0.0.1:8000/what-if' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @sample.json
```

#### Test Mortality Endpoints
```bash
# Predict Mortality
curl -X 'POST' \
  'http://127.0.0.1:8000/predict/mortality' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @sample.json

# Mortality What-If
curl -X 'POST' \
  'http://127.0.0.1:8000/what-if/mortality' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d @sample.json
```
