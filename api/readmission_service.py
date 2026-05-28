import os
import json
import joblib
import numpy as np

MODEL_PATH = "models/readmission_model.pkl"
METADATA_PATH = "models/metadata.json"

class ReadmissionService:
    """
    Service for predicting 30-day readmission risk using XGBoost.
    Uses label encoding mapped from training categories.
    """
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Readmission model file not found at {MODEL_PATH}")
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")

        self.model = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.features_order = self.metadata.get("features_order", [])
        self.discharge_location_mapping = self.metadata.get("discharge_location_mapping", {})

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
                "name": "HOME"
            },
            "HOME HEALTH CARE": {
                "code": self.discharge_location_mapping.get("HOME HEALTH CARE", 6),
                "name": "HOME HEALTH CARE"
            },
            "SKILLED NURSING FACILITY": {
                "code": self.discharge_location_mapping.get("SKILLED NURSING FACILITY", 11),
                "name": "SKILLED NURSING FACILITY"
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
