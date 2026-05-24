import joblib
import pandas as pd
import numpy as np
from mortality_service import MortalityService
import json

service = MortalityService()

# Tạo một bệnh nhân giả lập nhạy cảm (để mô hình phân biệt HOME và HOME HEALTH CARE)
np.random.seed(99)
dummy_patient = {feat: 0.0 for feat in service.features_order}
dummy_patient["age"] = 80.0
dummy_patient["gender"] = "M"
dummy_patient["duration_days"] = 2.0
dummy_patient["sbp_mean"] = 90.0
dummy_patient["bun_mean"] = 45.0


with open("dummy_patient.json", "w") as f:
    json.dump(dummy_patient, f, indent=2)

# Chạy thử What-if giả lập trên bệnh nhân ngẫu nhiên này
res = service.run_what_if_simulation(dummy_patient)
for key in ["HOME", "HOME HEALTH CARE", "SKILLED NURSING FACILITY"]:
    print(f"{key}: risk = {res[key]['mortality_risk_12m']:.6f}")
