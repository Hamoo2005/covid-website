import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "covid_model.pkl"
SELECTOR_PATH = BASE_DIR / "selector.pkl"
COLUMNS_PATH = BASE_DIR / "columns.pkl"

MODEL_INPUT_FIELDS = [
    "MEDICAL_UNIT",
    "PATIENT_TYPE",
    "INTUBED",
    "PNEUMONIA",
    "AGE",
    "DIABETES",
    "HIPERTENSION",
    "RENAL_CHRONIC",
    "CLASIFFICATION_FINAL",
    "ICU",
]

YES_NO_FIELDS = {
    "INTUBED",
    "PNEUMONIA",
    "DIABETES",
    "HIPERTENSION",
    "RENAL_CHRONIC",
    "ICU",
}

FIELD_LABELS = {
    "MEDICAL_UNIT": "Medical Unit",
    "PATIENT_TYPE": "Patient Type",
    "INTUBED": "Intubed",
    "PNEUMONIA": "Pneumonia",
    "AGE": "Age",
    "DIABETES": "Diabetes",
    "HIPERTENSION": "Hipertension",
    "RENAL_CHRONIC": "Renal Chronic",
    "CLASIFFICATION_FINAL": "COVID Classification",
    "ICU": "ICU",
}

app = Flask(__name__)
CORS(app)


def load_artifacts():
    missing_files = [
        path.name
        for path in (MODEL_PATH, SELECTOR_PATH, COLUMNS_PATH)
        if not path.exists()
    ]
    if missing_files:
        missing_list = ", ".join(missing_files)
        raise FileNotFoundError(
            f"Missing model artifact(s): {missing_list}. Run train_model.py once to generate them."
        )

    model = joblib.load(MODEL_PATH)
    selector = joblib.load(SELECTOR_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, selector, columns


try:
    MODEL, SELECTOR, ORIGINAL_COLUMNS = load_artifacts()
    MODEL_LOAD_ERROR = None
except Exception as exc:  # pragma: no cover - graceful startup behavior
    MODEL = None
    SELECTOR = None
    ORIGINAL_COLUMNS = None
    MODEL_LOAD_ERROR = str(exc)


def parse_int(value, field_name, *, allowed_values=None, minimum=None, maximum=None):
    label = FIELD_LABELS[field_name]

    if value is None or str(value).strip() == "":
        raise ValueError(f"{label} is required.")

    try:
        parsed = int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be a whole number.") from exc

    if allowed_values is not None and parsed not in allowed_values:
        allowed_text = ", ".join(str(item) for item in sorted(allowed_values))
        raise ValueError(f"{label} must be one of: {allowed_text}.")
    if minimum is not None and parsed < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    if maximum is not None and parsed > maximum:
        raise ValueError(f"{label} must be no more than {maximum}.")

    return parsed


def parse_yes_no(value, field_name):
    label = FIELD_LABELS[field_name]
    normalized = str(value).strip().lower()

    if normalized in {"yes", "1", "true"}:
        return 1
    if normalized in {"no", "0", "false"}:
        return 0

    raise ValueError(f"{label} must be Yes or No.")


def validate_medical_inputs(payload):
    if not isinstance(payload, dict):
        raise ValueError("medical_inputs must be a JSON object.")

    cleaned = {}
    for field in MODEL_INPUT_FIELDS:
        raw_value = payload.get(field)

        if field in YES_NO_FIELDS:
            cleaned[field] = parse_yes_no(raw_value, field)
        elif field == "PATIENT_TYPE":
            cleaned[field] = parse_int(raw_value, field, allowed_values={1, 2})
        elif field == "CLASIFFICATION_FINAL":
            cleaned[field] = parse_int(
                raw_value, field, allowed_values={1, 2, 3, 4, 5, 6, 7}
            )
        elif field == "AGE":
            cleaned[field] = parse_int(raw_value, field, minimum=0, maximum=120)
        else:
            cleaned[field] = parse_int(raw_value, field, minimum=1)

    return cleaned


@app.get("/")
def home():
    return jsonify(
        {
            "message": "COVID-19 risk prediction API is running.",
            "model_ready": MODEL_LOAD_ERROR is None,
            "error": MODEL_LOAD_ERROR,
        }
    )


@app.get("/health")
def health():
    status_code = 200 if MODEL_LOAD_ERROR is None else 503
    return (
        jsonify(
            {
                "status": "ok" if MODEL_LOAD_ERROR is None else "unavailable",
                "model_ready": MODEL_LOAD_ERROR is None,
                "error": MODEL_LOAD_ERROR,
            }
        ),
        status_code,
    )


@app.post("/predict")
def predict():
    if MODEL_LOAD_ERROR is not None:
        return jsonify({"error": MODEL_LOAD_ERROR}), 503

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    medical_payload = payload.get("medical_inputs", payload)

    try:
        cleaned_inputs = validate_medical_inputs(medical_payload)

        # Personal report fields are intentionally excluded from the ML pipeline.
        new_patient = pd.DataFrame([cleaned_inputs])
        new_patient_full = pd.DataFrame(0, index=[0], columns=ORIGINAL_COLUMNS)

        for column in new_patient.columns:
            if column in new_patient_full.columns:
                new_patient_full[column] = new_patient[column].values

        transformed_patient = SELECTOR.transform(new_patient_full)
        prediction = int(MODEL.predict(transformed_patient)[0])
        probabilities = MODEL.predict_proba(transformed_patient)[0]
        probability_map = dict(zip(MODEL.classes_, probabilities))

        survived_probability = round(float(probability_map.get(0, 0.0) * 100), 2)
        death_probability = round(float(probability_map.get(1, 0.0) * 100), 2)

        return jsonify(
            {
                "result": (
                    "High Risk (Death)"
                    if prediction == 1
                    else "Low Risk (Survived)"
                ),
                "survived_probability": survived_probability,
                "death_probability": death_probability,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Prediction failed. Please try again."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
