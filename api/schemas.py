import json
from typing import Optional, Union, Dict, List
from pydantic import BaseModel, create_model

# Path to metadata config
METADATA_PATH = "models/metadata.json"

try:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    features_order = metadata.get("features_order", [])
    discharge_location_mapping = metadata.get("discharge_location_mapping", {})
except Exception:
    features_order = []
    discharge_location_mapping = {}

# Dynamically construct the Pydantic fields dictionary
# We default most numerical features to 0.0 so the user doesn't have to fill in 236 fields.
# For discharge_location and gender, we allow string or integer inputs for ease of testing.
fields = {}
for feature in features_order:
    if feature == "discharge_location":
        fields[feature] = (Optional[Union[str, int]], "HOME")
    elif feature == "gender":
        fields[feature] = (Optional[Union[str, int]], "M")
    else:
        fields[feature] = (Optional[float], 0.0)

# Create the dynamic request model (shared for both readmission and mortality tasks)
PredictRequest = create_model(
    "PredictRequest",
    **fields,
    __doc__="Dynamic request schema containing all 236 patient features required by the readmission and mortality models."
)

# ----------------- Readmission Schemas -----------------
class Curve30Day(BaseModel):
    days: List[int]
    probabilities: List[float]

class PredictResponseData(BaseModel):
    prediction: int
    confidence: float
    readmission_probability: float
    curve_30day: Curve30Day

class PredictResponse(BaseModel):
    status: str
    data: PredictResponseData

class ScenarioDetail(BaseModel):
    name: str
    code: int
    readmission_probability: float
    curve_30day: Curve30Day

class WhatIfResponse(BaseModel):
    status: str
    data: Dict[str, ScenarioDetail]

# ------------------ Mortality Schemas ------------------
class SurvivalCurve(BaseModel):
    days: List[int]
    probabilities: List[float]

class PredictMortalityResponseData(BaseModel):
    mortality_risk_12m: float
    rmst_12m: float
    survival_curve: SurvivalCurve

class PredictMortalityResponse(BaseModel):
    status: str
    data: PredictMortalityResponseData

class ScenarioMortalityDetail(BaseModel):
    name: str
    code: int
    mortality_risk_12m: float
    rmst_12m: float
    survival_curve: SurvivalCurve

class WhatIfMortalityResponse(BaseModel):
    status: str
    data: Dict[str, ScenarioMortalityDetail]
