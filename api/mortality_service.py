import os
import json
import joblib
import numpy as np
import pandas as pd
import xgbse  # Required for joblib to unpickle the mortality model

MORTALITY_MODEL_PATH = "models/mortality_models_export/mortality_models/final_xgbse_model.joblib"
METADATA_PATH = "models/metadata.json"


class MortalityService:
    """
    Service for predicting 12-month mortality risk using XGBSE (survival ensemble).
    Uses shifted label encoding due to 'Missing' alphabetical category introduction.
    Only uses the final_xgbse model (no imputer).
    """
    def __init__(self):
        if not os.path.exists(MORTALITY_MODEL_PATH):
            raise FileNotFoundError(f"Mortality model file not found at {MORTALITY_MODEL_PATH}")
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")

        self.mortality_model = joblib.load(MORTALITY_MODEL_PATH)

        # Load features_order from metadata (which is identical to the mortality feature order)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.features_order = metadata.get("features_order", [])

        # Define separate mapping for Mortality Model (due to alphabetical sorting with "Missing" category shift)
        self.mortality_discharge_location_mapping = {
            "ACUTE HOSPITAL": 0,
            "AGAINST ADVICE": 1,
            "ASSISTED LIVING": 2,
            "CHRONIC/LONG TERM ACUTE CARE": 3,
            "HEALTHCARE FACILITY": 4,
            "HOME": 5,
            "HOME HEALTH CARE": 6,
            "HOSPICE": 7,
            "MISSING": 8,
            "KHÔNG XÁC ĐỊNH / KHUYẾT": 8,
            "OTHER FACILITY": 9,
            "PSYCH FACILITY": 10,
            "REHAB": 11,
            "SKILLED NURSING FACILITY": 12
        }

    def preprocess_features(self, payload_dict: dict) -> np.ndarray:
        features_list = []
        for feature in self.features_order:
            val = payload_dict.get(feature, 0.0)

            # 1. Handle discharge_location
            if feature == "discharge_location":
                if isinstance(val, str):
                    val_upper = val.upper()
                    val = self.mortality_discharge_location_mapping.get(
                        val_upper,
                        self.mortality_discharge_location_mapping.get("KHÔNG XÁC ĐỊNH / KHUYẾT", 8)
                    )
                elif val is None:
                    val = self.mortality_discharge_location_mapping.get("KHÔNG XÁC ĐỊNH / KHUYẾT", 8)

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

    def predict(self, payload_dict: dict) -> dict:
        features_array = self.preprocess_features(payload_dict)

        # Reconstruct DataFrame with feature names (needed for XGBSE / xgboost models trained with column names)
        X_df = pd.DataFrame(features_array, columns=self.features_order)

        # Predict survival probability S(t) directly using XGBSE model (no imputer)
        pred_survival = self.mortality_model.predict(X_df)

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

    def run_what_if_simulation(self, payload_dict: dict) -> dict:
        scenarios = {
            "HOME": {
                "code": 5,
                "name": "HOME"
            },
            "HOME HEALTH CARE": {
                "code": 6,
                "name": "HOME HEALTH CARE"
            },
            "SKILLED NURSING FACILITY": {
                "code": 12,
                "name": "SKILLED NURSING FACILITY"
            }
        }

        results = {}
        for key, info in scenarios.items():
            simulated_payload = payload_dict.copy()
            simulated_payload["discharge_location"] = info["code"]

            sim_result = self.predict(simulated_payload)
            results[key] = {
                "name": info["name"],
                "code": info["code"],
                "mortality_risk_12m": sim_result["mortality_risk_12m"],
                "rmst_12m": sim_result["rmst_12m"],
                "survival_curve": sim_result["survival_curve"]
            }
        return results
