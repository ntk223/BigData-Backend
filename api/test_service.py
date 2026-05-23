from api.services import ModelService
import json

service = ModelService()

# Create a sample payload from the schema default values
sample = {}
for feature in service.features_order:
    if feature == "age":
        sample[feature] = 75.0
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
    elif feature == "bun_mean":
        sample[feature] = 30.0
    elif feature == "pt_min":
        sample[feature] = 12.0
    elif feature.startswith("note_emb_"):
        sample[feature] = 0.01
    else:
        sample[feature] = 0.0

print("--- Testing run_what_if_simulation (Readmission) ---")
res_readmission = service.run_what_if_simulation(sample)
for key, val in res_readmission.items():
    print(f"Scenario {key} (code {val['code']}): prob = {val['readmission_probability']:.6f}")

print("\n--- Testing run_what_if_mortality_simulation (Mortality) ---")
res_mortality = service.run_what_if_mortality_simulation(sample)
for key, val in res_mortality.items():
    print(f"Scenario {key} (code {val['code']}): risk = {val['mortality_risk_12m']:.6f}")

