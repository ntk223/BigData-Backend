"""
XAI Explanation Service
-----------------------
Provides patient-level SHAP explanations for:
  - 30-day readmission risk (auxiliary XGBoost)
  - 12-month mortality risk (auxiliary XGBoost)

Uses pre-trained auxiliary models + SHAP TreeExplainer saved from notebooks.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

# ---------- paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

READMISSION_XAI_DIR = os.path.join(BASE_DIR, "xai_model_only_readmission")
MORTALITY_XAI_DIR   = os.path.join(BASE_DIR, "xai_model_only_mortality")

# ---------- feature display helpers ----------
HIDE_NOTE_EMBEDDINGS = True

FRIENDLY_NAMES = {
    "age": "Tuổi",
    "gender": "Giới tính",
    "admission_type": "Loại nhập viện",
    "insurance": "Bảo hiểm",
    "marital_status": "Tình trạng hôn nhân",
    "race": "Dân tộc",
    "discharge_location": "Phương án xuất viện",
    "duration_days": "Số ngày nằm viện",
    "sbp_mean": "Huyết áp tâm thu TB",
    "sbp_min": "Huyết áp tâm thu Min",
    "sbp_max": "Huyết áp tâm thu Max",
    "sbp_count": "Số lần đo huyết áp",
    "spo2_mean": "SpO2 trung bình",
    "spo2_count": "Số lần đo SpO2",
    "hr_mean": "Nhịp tim trung bình",
    "hr_count": "Số lần đo nhịp tim",
    "temperature_mean": "Nhiệt độ trung bình",
    "temperature_count": "Số lần đo nhiệt độ",
    "albumin_max": "Albumin Max",
    "albumin_mean": "Albumin TB",
    "albumin_min": "Albumin Min",
    "alt_max": "ALT Max",
    "alt_mean": "ALT TB",
    "alt_min": "ALT Min",
    "anion_gap_max": "Anion Gap Max",
    "anion_gap_mean": "Anion Gap TB",
    "anion_gap_min": "Anion Gap Min",
    "ast_max": "AST Max",
    "ast_mean": "AST TB",
    "ast_min": "AST Min",
    "bicarbonate_max": "Bicarbonate Max",
    "bicarbonate_mean": "Bicarbonate TB",
    "bicarbonate_min": "Bicarbonate Min",
    "bilirubin_total_max": "Bilirubin Max",
    "bilirubin_total_mean": "Bilirubin TB",
    "bilirubin_total_min": "Bilirubin Min",
    "bun_max": "BUN Max",
    "bun_mean": "BUN TB",
    "bun_min": "BUN Min",
    "calcium_max": "Calcium Max",
    "calcium_mean": "Calcium TB",
    "calcium_min": "Calcium Min",
    "chloride_max": "Chloride Max",
    "chloride_mean": "Chloride TB",
    "chloride_min": "Chloride Min",
    "creatinine_max": "Creatinine Max",
    "creatinine_mean": "Creatinine TB",
    "creatinine_min": "Creatinine Min",
    "glucose_max": "Glucose Max",
    "glucose_mean": "Glucose TB",
    "glucose_min": "Glucose Min",
    "hematocrit_max": "Hematocrit Max",
    "hematocrit_mean": "Hematocrit TB",
    "hematocrit_min": "Hematocrit Min",
    "hemoglobin_max": "Hemoglobin Max",
    "hemoglobin_mean": "Hemoglobin TB",
    "hemoglobin_min": "Hemoglobin Min",
    "inr_max": "INR Max",
    "inr_mean": "INR TB",
    "inr_min": "INR Min",
    "lactate_max": "Lactate Max",
    "lactate_mean": "Lactate TB",
    "lactate_min": "Lactate Min",
    "magnesium_max": "Magnesium Max",
    "magnesium_mean": "Magnesium TB",
    "magnesium_min": "Magnesium Min",
    "phosphate_max": "Phosphate Max",
    "phosphate_mean": "Phosphate TB",
    "phosphate_min": "Phosphate Min",
    "platelet_max": "Tiểu cầu Max",
    "platelet_mean": "Tiểu cầu TB",
    "platelet_min": "Tiểu cầu Min",
    "potassium_max": "Kali Max",
    "potassium_mean": "Kali TB",
    "potassium_min": "Kali Min",
    "pt_max": "PT Max",
    "pt_mean": "PT TB",
    "pt_min": "PT Min",
    "ptt_max": "PTT Max",
    "ptt_mean": "PTT TB",
    "ptt_min": "PTT Min",
    "sodium_max": "Natri Max",
    "sodium_mean": "Natri TB",
    "sodium_min": "Natri Min",
    "wbc_max": "Bạch cầu Max",
    "wbc_mean": "Bạch cầu TB",
    "wbc_min": "Bạch cầu Min",
}


def _strip_prefix(feature: str) -> str:
    """num__age -> age, cat__discharge_location_SNF -> discharge_location_SNF"""
    return feature.split("__", 1)[1] if "__" in feature else feature


def _parse_feature(feature: str):
    f = _strip_prefix(feature)
    if f.startswith("discharge_location_"):
        return "discharge_location", f.replace("discharge_location_", "")
    if f.startswith("gender_"):
        return "gender", f.replace("gender_", "")
    return f, None


def _is_note_embedding(feature: str) -> bool:
    f = _strip_prefix(feature)
    return f.startswith("note_emb_") or "note_emb_" in f


def _is_displayable(feature: str) -> bool:
    if HIDE_NOTE_EMBEDDINGS and _is_note_embedding(feature):
        return False
    return True


def _display_name(feature: str) -> str:
    base, category = _parse_feature(feature)
    if base == "discharge_location":
        return f"Phương án xuất viện: {category}" if category else "Phương án xuất viện"
    if base == "gender":
        return f"Giới tính: {category}" if category else "Giới tính"
    if base.startswith("icd10_chap_") or base.startswith("icd_chap_"):
        cleaned = base.replace("icd10_chap_", "").replace("icd_chap_", "").replace("_", " ").title()
        return f"ICD-10: {cleaned}"
    return FRIENDLY_NAMES.get(base, base)


def _importance_label(abs_val: float, all_abs: list) -> str:
    if not all_abs:
        return "low"
    p75 = float(np.percentile(all_abs, 75))
    p90 = float(np.percentile(all_abs, 90))
    if abs_val >= p90:
        return "high"
    elif abs_val >= p75:
        return "medium"
    return "low"


def _make_card(feature: str, value: float, shap_val: float, all_abs: list) -> dict:
    return {
        "feature": _strip_prefix(feature),
        "display_name": _display_name(feature),
        "value": round(float(value), 4),
        "shap_value": round(float(shap_val), 6),
        "direction": "increase_risk" if shap_val > 0 else "decrease_risk",
        "importance": _importance_label(abs(shap_val), all_abs),
    }


class XAIExplainer:
    """
    Loads xAI auxiliary model artifacts and provides per-patient SHAP explanations.
    """

    def __init__(self, xai_dir: str, task: str):
        import shap

        self.task = task
        self.xai_dir = xai_dir

        model_path = os.path.join(xai_dir, "xai_auxiliary_model.joblib")
        preprocessor_path = os.path.join(xai_dir, "xai_preprocessor.joblib")
        feature_cols_path = os.path.join(xai_dir, "feature_cols_raw.json")
        feature_names_path = os.path.join(xai_dir, "feature_names_processed.json")
        metrics_path = os.path.join(xai_dir, "xai_metrics.json")

        for p in [model_path, preprocessor_path, feature_cols_path, feature_names_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"XAI artifact not found: {p}")

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

        # Create a fresh SHAP TreeExplainer from the loaded model instead of loading
        # the pickled explainer, to avoid numba/Python version mismatch issues.
        self.explainer = shap.TreeExplainer(self.model)

        with open(feature_cols_path, "r", encoding="utf-8") as f:
            self.feature_cols = json.load(f)

        with open(feature_names_path, "r", encoding="utf-8") as f:
            self.feature_names_processed = json.load(f)

        self.metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

    def _build_raw_df(self, payload_dict: dict) -> pd.DataFrame:
        """Build a single-row DataFrame with the raw feature columns."""
        row = {}
        for col in self.feature_cols:
            val = payload_dict.get(col, 0.0)
            if val is None:
                val = 0.0
            # Keep string for categorical (discharge_location, gender, etc.)
            if col in ("discharge_location", "gender", "admission_type", "insurance",
                       "marital_status", "race"):
                row[col] = val if isinstance(val, str) else str(val) if val else ""
            else:
                try:
                    row[col] = float(val)
                except (TypeError, ValueError):
                    row[col] = 0.0
        return pd.DataFrame([row])

    def _transform(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        X_raw = raw_df[self.feature_cols].copy()
        X_processed = self.preprocessor.transform(X_raw)
        return pd.DataFrame(X_processed, columns=self.feature_names_processed)

    def _get_shap_values(self, X_processed: pd.DataFrame) -> np.ndarray:
        vals = self.explainer.shap_values(X_processed)
        if isinstance(vals, list):
            vals = vals[1]   # class-1 SHAP values
        if hasattr(vals, "values"):
            vals = vals.values
        return np.asarray(vals)

    def explain(self, payload_dict: dict, top_k: int = 5) -> dict:
        """
        Return a human-readable SHAP explanation for one patient.
        """
        raw_df = self._build_raw_df(payload_dict)
        Xp = self._transform(raw_df)

        # Risk probability from auxiliary model
        risk = float(self.model.predict_proba(Xp)[0, 1])

        # SHAP
        shap_row = self._get_shap_values(Xp)[0]

        # Collect all abs SHAP for displayable features (for importance labeling)
        all_abs = [
            abs(v) for f, v in zip(self.feature_names_processed, shap_row)
            if _is_displayable(f)
        ]

        cards = []
        for feature, value, sv in zip(self.feature_names_processed, Xp.iloc[0], shap_row):
            if _is_displayable(feature):
                cards.append(_make_card(feature, value, sv, all_abs))

        # Sort by absolute SHAP value
        cards.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        top_risk = [c for c in cards if c["direction"] == "increase_risk"][:top_k]
        top_protective = [c for c in cards if c["direction"] == "decrease_risk"][:top_k]

        risk_key = "risk_readmission_30d" if self.task == "readmission" else "risk_mortality_12m"
        target_name = self.metrics.get("target_name", self.task)

        return {
            "task": self.task,
            "target": target_name,
            risk_key: round(risk, 4),
            "risk_percent": round(risk * 100, 1),
            "top_risk_factors": top_risk,
            "top_protective_factors": top_protective,
            "note": "",
        }


class XAIService:
    """
    Unified service exposing readmission + mortality xAI explanations.
    """

    def __init__(self):
        self.readmission_explainer = None
        self.mortality_explainer = None

        if os.path.isdir(READMISSION_XAI_DIR):
            try:
                self.readmission_explainer = XAIExplainer(READMISSION_XAI_DIR, "readmission")
                print("[XAI] Readmission explainer loaded ✓")
            except Exception as e:
                print(f"[XAI] Failed to load readmission explainer: {e}")

        if os.path.isdir(MORTALITY_XAI_DIR):
            try:
                self.mortality_explainer = XAIExplainer(MORTALITY_XAI_DIR, "mortality")
                print("[XAI] Mortality explainer loaded ✓")
            except Exception as e:
                print(f"[XAI] Failed to load mortality explainer: {e}")

    def explain_readmission(self, payload_dict: dict, top_k: int = 5) -> dict:
        if not self.readmission_explainer:
            raise RuntimeError("Readmission XAI explainer is not available.")
        return self.readmission_explainer.explain(payload_dict, top_k)

    def explain_mortality(self, payload_dict: dict, top_k: int = 5) -> dict:
        if not self.mortality_explainer:
            raise RuntimeError("Mortality XAI explainer is not available.")
        return self.mortality_explainer.explain(payload_dict, top_k)
