# Audit Memo - Yogyank Entitlement Score Baseline Review

## Summary of Risks Identified

The original baseline training script contained several major risks that made the reported validation performance unreliable for real-world scoring use. The most critical issue was severe data leakage through the feature `defaulted_in_next_12_months`, which represents future information unavailable at scoring time. Correlation analysis showed this feature had a very strong relationship with the target score (-0.88), strongly suggesting that the original R² value was artificially inflated. The script also directly modified target scores based on `pm_kisan_status`, which improperly mixed business policy logic into the model training process. In addition, the original implementation used unsafe categorical encoding with `LabelEncoder`, performed randomized validation splitting that did not simulate future scoring behavior, and saved only the model artifact without preprocessing metadata, schema definitions, or reproducibility controls.

## Changes Implemented

The revised pipeline removes the future-leakage feature `defaulted_in_next_12_months` and excludes identifier-only fields such as `farmer_id`. Policy logic was separated from model training by removing manual score adjustments tied to PM Kisan status. Validation was changed to an out-of-time split using `application_year`, where older years were used for training and 2024 data was reserved for validation in order to better simulate future scoring conditions. A leakage-safe sklearn Pipeline with `ColumnTransformer` and `OneHotEncoder(handle_unknown="ignore")` was implemented to ensure deterministic preprocessing and inference consistency. The updated solution also saves the full pipeline artifact, feature schema, metrics, version metadata, and sample reason codes for reproducibility and audit traceability.

## Limitations

This assessment used a synthetic dataset with limited temporal depth, so the current validation setup should not be considered sufficient for production deployment. The reason-code implementation is currently based on global feature importance and should be replaced with stronger local explainability methods such as SHAP before real deployment. Additional fairness testing, drift monitoring, and challenger-model evaluation would also be recommended.

## Monitoring Recommendations

The following monitoring slices should be tracked during deployment and periodic model review:

* PM Kisan status segments
* District-wise score distribution drift
* Crop-type performance stability
* Historical repayment-score distribution shifts
* Annual application-year drift
* Prediction stability across irrigation types
