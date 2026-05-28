# LLM_NOTES.md

## Tools Used

* ChatGPT

---

# Where AI Assistance Was Used

AI assistance was used for:

* identifying leakage and validation risks
* discussing safer preprocessing approaches
* restructuring the training script into a cleaner pipeline architecture
* improving auditability and artifact persistence
* drafting documentation structure
* reviewing explainability and reason-code approaches

---

# Representative Prompts

1. Review the baseline training script for leakage, validation, and auditability issues.

2. Explain why LabelEncoder is unsafe for nominal categorical variables in this use case.

3. Suggest a safer sklearn pipeline structure for preprocessing and model training.

4. Explain why out-of-time validation is preferable over random train/test splitting for future scoring systems.

5. Suggest lightweight explainability approaches suitable for a 90-minute assessment.

---

# Suggestions Accepted

* Replaced LabelEncoder with OneHotEncoder inside a sklearn Pipeline
* Converted validation from random split to out-of-time split using application_year
* Added artifact persistence including schema, metrics, and version metadata
* Added lightweight reason-code generation using feature importance

---

# Suggestions Rejected or Corrected

* Rejected overly complex explainability approaches that would reduce audit clarity within the assessment time limit
* Avoided adding unnecessary hyperparameter tuning focused only on improving R² without improving validation trustworthiness

---

# Personally Verified

The following items were manually reviewed and verified:

* leakage removal for `defaulted_in_next_12_months`
* out-of-time validation split behavior
* preprocessing pipeline fit only on training data
* saved artifact generation
* reason-code output generation
* successful end-to-end script execution
* realistic post-leakage validation performance