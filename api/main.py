from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import (
    PredictRequest, 
    PredictResponse, 
    WhatIfResponse,
    PredictMortalityResponse,
    WhatIfMortalityResponse
)
from api.readmission_service import ReadmissionService
from api.mortality_service import MortalityService

# Initialize the model services
try:
    readmission_service = ReadmissionService()
except Exception as e:
    print(f"Error initializing ReadmissionService: {e}")
    readmission_service = None

try:
    mortality_service = MortalityService()
except Exception as e:
    print(f"Error initializing MortalityService: {e}")
    mortality_service = None

# Initialize FastAPI App
app = FastAPI(
    title="PREDICTCARE AI - Backend CDSS API (Team 10)",
    description="REST API for predicting 30-day readmission risk (XGBoost) and 12-month mortality risk (XGBSE).",
    version="1.2.0",
)

# Add CORS Middleware to allow cross-origin testing/frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "project": "PREDICTCARE AI - CDSS Dashboard (Team 10)",
        "tasks": [
            "30-day Readmission Risk Prediction",
            "12-month Mortality Risk Prediction"
        ],
        "endpoints": {
            "swagger_docs": "/docs",
            "metadata": "/metadata",
            "sample_request": "/sample",
            "predict_readmission": "/predict/",
            "what_if_readmission": "/what-if",
            "predict_mortality": "/predict/mortality",
            "what_if_mortality": "/what-if/mortality"
        }
    }

@app.get("/metadata")
def get_metadata():
    if not readmission_service:
        raise HTTPException(status_code=503, detail="Readmission service is currently unavailable.")
    return readmission_service.metadata

@app.get("/sample")
def get_sample_payload():
    if not readmission_service:
        raise HTTPException(status_code=503, detail="Readmission service is currently unavailable.")
        
    sample = {}
    for feature in readmission_service.features_order:
        if feature == "age":
            sample[feature] = 65.0
        elif feature == "gender":
            sample[feature] = "M"
        elif feature == "discharge_location":
            sample[feature] = "HOME"
        elif feature == "duration_days":
            sample[feature] = 4.0
        elif feature == "sbp_mean":
            sample[feature] = 120.0
        elif feature == "spo2_mean":
            sample[feature] = 97.5
        elif feature == "hr_mean":
            sample[feature] = 80.0
        elif feature == "temperature_mean":
            sample[feature] = 37.0
        elif feature.startswith("note_emb_"):
            sample[feature] = 0.01
        else:
            sample[feature] = 0.0
            
    return sample

# ----------------- Readmission Routes -----------------
@app.post("/predict/readmission", response_model=PredictResponse)
async def predict_readmission(payload: PredictRequest):
    if not readmission_service:
        raise HTTPException(status_code=503, detail="Readmission service is currently unavailable.")
    try:
        payload_dict = payload.model_dump()
        result = readmission_service.predict(payload_dict)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/what-if/readmission", response_model=WhatIfResponse)
async def what_if_simulation(payload: PredictRequest):
    if not readmission_service:
        raise HTTPException(status_code=503, detail="Readmission service is currently unavailable.")
    try:
        payload_dict = payload.model_dump()
        result = readmission_service.run_what_if_simulation(payload_dict)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"What-If simulation error: {str(e)}")

# ------------------ Mortality Routes ------------------
@app.post("/predict/mortality", response_model=PredictMortalityResponse)
async def predict_mortality(payload: PredictRequest):
    if not mortality_service:
        raise HTTPException(status_code=503, detail="Mortality service is currently unavailable.")
    try:
        payload_dict = payload.model_dump()
        result = mortality_service.predict(payload_dict)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mortality prediction error: {str(e)}")

@app.post("/what-if/mortality", response_model=WhatIfMortalityResponse)
async def what_if_mortality_simulation(payload: PredictRequest):
    if not mortality_service:
        raise HTTPException(status_code=503, detail="Mortality service is currently unavailable.")
    try:
        payload_dict = payload.model_dump()
        result = mortality_service.run_what_if_simulation(payload_dict)
        return {
            "status": "success",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"What-If mortality simulation error: {str(e)}")
