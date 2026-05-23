import os
import json
import joblib
import numpy as np
import pandas as pd
import xgbse  # Required for joblib to unpickle the mortality model

MODEL_PATH = "models/readmission_model.pkl"
METADATA_PATH = "models/metadata.json"

MORTALITY_MODEL_PATH = "models/mortality_models_export/mortality_models/final_xgbse_model.joblib"
MORTALITY_IMPUTER_PATH = "models/mortality_models_export/mortality_models/fitted_imputer.joblib"

class ModelService:
    def __init__(self):
        # 1. Load Readmission Model & Metadata
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Readmission model file not found at {MODEL_PATH}")
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")
            
        self.model = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            
        self.features_order = self.metadata.get("features_order", [])
        self.discharge_location_mapping = self.metadata.get("discharge_location_mapping", {})
        
        # 2. Load Mortality Model & Imputer
        if not os.path.exists(MORTALITY_MODEL_PATH):
            raise FileNotFoundError(f"Mortality model file not found at {MORTALITY_MODEL_PATH}")
        if not os.path.exists(MORTALITY_IMPUTER_PATH):
            raise FileNotFoundError(f"Mortality imputer file not found at {MORTALITY_IMPUTER_PATH}")
            
        self.mortality_model = joblib.load(MORTALITY_MODEL_PATH)
        self.mortality_imputer = joblib.load(MORTALITY_IMPUTER_PATH)

    def preprocess_features(self, payload_dict: dict) -> np.ndarray:
        features_list = []
        for feature in self.features_order:
            val = payload_dict.get(feature, 0.0)
            
            # 1. Handle discharge_location
            if feature == "discharge_location":
                if isinstance(val, str):
                    val_upper = val.upper()
                    val = self.discharge_location_mapping.get(
                        val_upper, 
                        self.discharge_location_mapping.get("KHÔNG XÁC ĐỊNH / KHUYẾT", 12)
                    )
                elif val is None:
                    val = self.discharge_location_mapping.get("KHÔNG XÁC ĐỊNH / KHUYẾT", 12)
            
            # 2. Handle gender mapping
            elif feature == "gender":
                if isinstance(val, str):
                    gender_mapping = {
                        'M': 1, 'F': 0, 'MALE': 1, 'FEMALE': 0,
                        '1': 1, '0': 0
                    }
                    val = gender_mapping.get(val.upper(), 0)
                elif val is None:
                    val = 0
            
            if val is None:
                val = 0.0
                
            features_list.append(float(val))
            
        return np.array([features_list])

    # ----------------- Readmission Methods -----------------
    def predict(self, payload_dict: dict) -> dict:
        features_array = self.preprocess_features(payload_dict)
        prediction = int(self.model.predict(features_array)[0])
        probabilities = self.model.predict_proba(features_array)[0]
        confidence = float(probabilities[prediction])
        
        # Calculate the 30-day readmission curve
        p_total = float(probabilities[1])
        days, p_series = self.generate_30day_readmission_curve(p_total)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "readmission_probability": p_total,
            "curve_30day": {
                "days": days.tolist(),
                "probabilities": p_series.tolist()
            }
        }

    def generate_30day_readmission_curve(self, p_total: float, gamma: float = 0.08):
        days = np.arange(0, 31, 1)
        distribution = (1 - np.exp(-gamma * days)) / (1 - np.exp(-gamma * 30))
        return days, (p_total * distribution)

    def run_what_if_simulation(self, payload_dict: dict) -> dict:
        scenarios = {
            "HOME": {
                "code": self.discharge_location_mapping.get("HOME", 5),
                "name": "Về nhà (HOME) - Tự chăm sóc"
            },
            "HOME HEALTH CARE": {
                "code": self.discharge_location_mapping.get("HOME HEALTH CARE", 6),
                "name": "HOME HEALTH CARE - Có điều dưỡng hỗ trợ"
            },
            "SKILLED NURSING FACILITY": {
                "code": self.discharge_location_mapping.get("SKILLED NURSING FACILITY", 11),
                "name": "VIỆN ĐIỀU DƯỠNG (SNF) - Chăm sóc 24/7"
            }
        }
        
        results = {}
        for key, info in scenarios.items():
            simulated_payload = payload_dict.copy()
            simulated_payload["discharge_location"] = info["code"]
            
            features_array = self.preprocess_features(simulated_payload)
            probabilities = self.model.predict_proba(features_array)[0]
            p_total = float(probabilities[1])
            
            days, p_series = self.generate_30day_readmission_curve(p_total)
            results[key] = {
                "name": info["name"],
                "code": info["code"],
                "readmission_probability": p_total,
                "curve_30day": {
                    "days": days.tolist(),
                    "probabilities": p_series.tolist()
                }
            }
        return results

    # ------------------ Mortality Methods ------------------
    def predict_mortality(self, payload_dict: dict) -> dict:
        features_array = self.preprocess_features(payload_dict)
        
        # Reconstruct DataFrame with feature names (needed for XGBSE / xgboost models trained with column names)
        X_df = pd.DataFrame(features_array, columns=self.features_order)
        
        # Apply imputer transformation
        X_imputed = pd.DataFrame(self.mortality_imputer.transform(X_df), columns=self.features_order)
        
        # Predict survival probability S(t)
        pred_survival = self.mortality_model.predict(X_imputed)
        
        # Extract timeline and probability values
        times = pred_survival.columns.astype(float)
        probabilities = pred_survival.iloc[0].values
        
        # Insert time 0 with S(0) = 1.0
        all_times = np.insert(times, 0, 0.0)
        all_probs = np.insert(probabilities, 0, 1.0)
        
        # Mortality risk at 12 months = 1 - S(365)
        # S(365) is the last element in the probabilities array
        risk_12m = float(1.0 - probabilities[-1])
        
        # RMST (Restricted Mean Survival Time) using trapezoidal rule (Area Under Survival Curve)
        rmst_12m = float(np.trapz(all_probs, all_times))
        
        return {
            "mortality_risk_12m": risk_12m,
            "rmst_12m": rmst_12m,
            "survival_curve": {
                "days": all_times.astype(int).tolist(),
                "probabilities": all_probs.tolist()
            }
        }

    def run_what_if_mortality_simulation(self, payload_dict: dict) -> dict:
        scenarios = {
            "HOME": {
                "code": self.discharge_location_mapping.get("HOME", 5),
                "name": "Về nhà (HOME) - Tự chăm sóc"
            },
            "HOME HEALTH CARE": {
                "code": self.discharge_location_mapping.get("HOME HEALTH CARE", 6),
                "name": "HOME HEALTH CARE - Có điều dưỡng hỗ trợ"
            },
            "SKILLED NURSING FACILITY": {
                "code": self.discharge_location_mapping.get("SKILLED NURSING FACILITY", 11),
                "name": "VIỆN ĐIỀU DƯỠNG (SNF) - Chăm sóc 24/7"
            }
        }
        
        results = {}
        for key, info in scenarios.items():
            simulated_payload = payload_dict.copy()
            simulated_payload["discharge_location"] = info["code"]
            
            sim_result = self.predict_mortality(simulated_payload)
            results[key] = {
                "name": info["name"],
                "code": info["code"],
                "mortality_risk_12m": sim_result["mortality_risk_12m"],
                "rmst_12m": sim_result["rmst_12m"],
                "survival_curve": sim_result["survival_curve"]
            }
        return results
