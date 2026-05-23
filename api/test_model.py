import joblib
import json
import numpy as np

model = joblib.load("models/readmission_model.pkl")
with open("models/metadata.json", "r") as f:
    metadata = json.load(f)

print("Features order:", metadata["features_order"][:20])
print("Discharge location mapping:", metadata["discharge_location_mapping"])
print("Model type:", type(model))

# Let's see if discharge_location is indeed in features_order
discharge_location_idx = metadata["features_order"].index("discharge_location")
print("discharge_location index in features_order:", discharge_location_idx)

# Let's test predictions with different discharge locations
# We'll create a baseline vector of all zeros, except age=60, gender=1, sbp_mean=120, etc.
base_vector = [0.0] * len(metadata["features_order"])
base_vector[metadata["features_order"].index("age")] = 60.0
base_vector[metadata["features_order"].index("gender")] = 1.0

# HOME (5), HOME HEALTH CARE (6), SKILLED NURSING FACILITY (11)
for loc in [5, 6, 11]:
    vec = list(base_vector)
    vec[discharge_location_idx] = float(loc)
    X = np.array([vec])
    probs = model.predict_proba(X)[0]
    print(f"Location {loc} -> prob of readmission: {probs[1]:.6f}")
