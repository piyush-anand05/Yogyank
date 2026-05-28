# README.md

## Yogyank Entitlement Score - Round 1 Technical Assessment

### Candidate

Piyush Anand

### Assessment Timing

* Start Time: 28 May 2026, 12:00 PM
* Approximate Time Spent: ~90 minutes

---

# Objective

The goal of this assessment was to audit and improve a baseline entitlement-score training pipeline for safer and more audit-conscious future scoring.

The revised implementation prioritizes:

* leakage prevention
* realistic validation
* auditability
* reproducibility
* explainability
* deterministic preprocessing

over maximizing headline validation metrics.

---

# Key Issues Identified in Original Script

The original baseline implementation contained several major risks:

* Future leakage through `defaulted_in_next_12_months`
* Manual modification of target values using PM Kisan policy logic
* Unsafe categorical encoding using `LabelEncoder`
* Randomized validation split not representative of future scoring
* Missing preprocessing artifact persistence
* No schema/version tracking
* No explainability or reason-code mechanism

---

# Improvements Implemented

## Leakage Control

* Removed `defaulted_in_next_12_months` from training features
* Removed manual target manipulation tied to PM Kisan status

## Validation Improvements

* Replaced random train/test split with out-of-time validation using `application_year`

## Safer Preprocessing

* Implemented sklearn `Pipeline`
* Added `ColumnTransformer`
* Used `OneHotEncoder(handle_unknown="ignore")`
* Added imputers for future robustness

## Auditability

* Saved full preprocessing + model pipeline
* Saved metrics
* Saved schema definitions
* Saved version metadata
* Saved sample reason codes

## Explainability

* Implemented lightweight top-3 reason code generation using feature importance

---

# Validation Results

* R2 Score: 0.3357
* MAE: 78.64

The reduced validation performance compared to the original script is expected after removing future-information leakage and implementing more realistic validation methodology.

---

# Artifacts Generated

The following artifacts are saved under `artifacts/`:

* `yogyank_pipeline.pkl`
* `metrics.json`
* `feature_schema.json`
* `version.json`
* `sample_reason_codes.json`

---

# Setup Instructions

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run training pipeline

```bash
python fixed_yogyank_training.py
```

---

# Assumptions

* `application_year` can be used as a proxy temporal split column
* PM Kisan status may remain as a model feature but should not directly alter targets
* Current dataset is synthetic and simplified for assessment purposes

---

# Limitations

* Current explainability uses global feature importance rather than local explanations such as SHAP
* Validation setup uses a single out-of-time split only
* No drift monitoring or fairness dashboard implemented due to time constraints

---

# Items Skipped Due to Time Constraints

* SHAP-based local explainability
* Automated data drift checks
* Fairness metric computation
* Hyperparameter optimization
* Unit/integration tests
