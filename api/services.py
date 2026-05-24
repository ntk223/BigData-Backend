from api.readmission_service import ReadmissionService
from api.mortality_service import MortalityService

class ModelService:
    """
    Backward compatibility facade that aggregates both ReadmissionService and MortalityService.
    """
    def __init__(self):
        self.readmission_service = ReadmissionService()
        self.mortality_service = MortalityService()
        self.features_order = self.readmission_service.features_order
        self.metadata = self.readmission_service.metadata
        self.discharge_location_mapping = self.readmission_service.discharge_location_mapping
        self.mortality_discharge_location_mapping = self.mortality_service.mortality_discharge_location_mapping

    def preprocess_features(self, payload_dict: dict, is_mortality: bool = False):
        if is_mortality:
            return self.mortality_service.preprocess_features(payload_dict)
        else:
            return self.readmission_service.preprocess_features(payload_dict)

    def predict(self, payload_dict: dict):
        return self.readmission_service.predict(payload_dict)

    def generate_30day_readmission_curve(self, p_total: float, gamma: float = 0.08):
        return self.readmission_service.generate_30day_readmission_curve(p_total, gamma)

    def run_what_if_simulation(self, payload_dict: dict):
        return self.readmission_service.run_what_if_simulation(payload_dict)

    def predict_mortality(self, payload_dict: dict):
        return self.mortality_service.predict(payload_dict)

    def run_what_if_mortality_simulation(self, payload_dict: dict):
        return self.mortality_service.run_what_if_simulation(payload_dict)
