import os
from pathlib import Path
import logging

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Covid Data.csv"
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INCLUDE_PREDICTION_DEBUG = os.environ.get("INCLUDE_PREDICTION_DEBUG", "1") == "1"


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


def get_selected_features(selector, columns):
    if selector is None or columns is None:
        return []

    support = getattr(selector, "get_support", None)
    if support is None:
        return []

    return [column for column, keep in zip(columns, selector.get_support()) if keep]


def load_training_debug_info():
    debug_info = {
        "y_value_counts": {},
        "selected_features": [],
    }

    if MODEL_LOAD_ERROR is not None:
        return debug_info

    debug_info["selected_features"] = get_selected_features(SELECTOR, ORIGINAL_COLUMNS)

    if not DATA_PATH.exists():
        return debug_info

    try:
        df = pd.read_csv(DATA_PATH)
        df["DEATH"] = df["DATE_DIED"].apply(lambda x: 0 if x == "9999-99-99" else 1)
        df = df.drop(columns=["DATE_DIED"])
        df = df.replace(2, 0)
        cols_to_clean = df.columns.drop("AGE")
        df[cols_to_clean] = df[cols_to_clean].replace([97, 98, 99], 0)
        debug_info["y_value_counts"] = {
            str(key): int(value) for key, value in df["DEATH"].value_counts().to_dict().items()
        }
    except Exception as exc:  # pragma: no cover - debug fallback
        logger.warning("Unable to load training debug info: %s", exc)

    return debug_info


def run_self_test():
    if MODEL_LOAD_ERROR is not None:
        return {"status": "unavailable", "error": MODEL_LOAD_ERROR}

    test_case = {
        "MEDICAL_UNIT": 1,
        "PATIENT_TYPE": 2,
        "INTUBED": 1,
        "PNEUMONIA": 1,
        "AGE": 80,
        "DIABETES": 1,
        "HIPERTENSION": 1,
        "RENAL_CHRONIC": 1,
        "CLASIFFICATION_FINAL": 1,
        "ICU": 1,
    }

    new_patient = pd.DataFrame([test_case])
    new_patient_full = pd.DataFrame(0, index=[0], columns=ORIGINAL_COLUMNS)

    for column in new_patient.columns:
        if column in new_patient_full.columns:
            new_patient_full[column] = new_patient[column].values

    transformed_patient = SELECTOR.transform(new_patient_full)
    prediction = int(MODEL.predict(transformed_patient)[0])
    probabilities = MODEL.predict_proba(transformed_patient)[0]

    return {
        "status": "ok",
        "test_case": test_case,
        "prediction": prediction,
        "result": "High Risk (Death)" if prediction == 1 else "Low Risk (Survived)",
        "probabilities_by_class": {
            str(int(cls)): round(float(prob) * 100, 2)
            for cls, prob in zip(MODEL.classes_, probabilities)
        },
        "first_transformed_input": transformed_patient[0].tolist(),
    }


try:
    MODEL, SELECTOR, ORIGINAL_COLUMNS = load_artifacts()
    MODEL_LOAD_ERROR = None
except Exception as exc:  # pragma: no cover - graceful startup behavior
    MODEL = None
    SELECTOR = None
    ORIGINAL_COLUMNS = None
    MODEL_LOAD_ERROR = str(exc)

SELECTED_FEATURES = get_selected_features(SELECTOR, ORIGINAL_COLUMNS)
MISSING_SELECTED_FEATURES_FROM_UI = [
    feature for feature in SELECTED_FEATURES if feature not in MODEL_INPUT_FIELDS
]
TRAINING_DEBUG_INFO = load_training_debug_info()
SELF_TEST_RESULT = run_self_test()

if MODEL_LOAD_ERROR is None:
    logger.info("Selected features: %s", SELECTED_FEATURES)
    logger.info("Target distribution: %s", TRAINING_DEBUG_INFO.get("y_value_counts", {}))
    logger.info("Model classes: %s", [int(cls) for cls in MODEL.classes_.tolist()])
    logger.info("Startup self-test: %s", SELF_TEST_RESULT)
else:
    logger.error("Model failed to load: %s", MODEL_LOAD_ERROR)


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
                "self_test": SELF_TEST_RESULT,
            }
        ),
        status_code,
    )


@app.post("/predict")
def predict():
    if MODEL_LOAD_ERROR is not None:
        return jsonify({"error": MODEL_LOAD_ERROR}), 503

    raw_body = request.get_data(as_text=True)
    payload = request.get_json(silent=True)
    logger.info("Received /predict raw body: %s", raw_body)
    logger.info("Received /predict parsed JSON: %s", payload)

    if not payload:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    medical_payload = payload.get("medical_inputs", payload)

    try:
        cleaned_inputs = validate_medical_inputs(medical_payload)
        logger.info("Processed medical inputs: %s", cleaned_inputs)

        # Personal report fields are intentionally excluded from the ML pipeline.
        new_patient = pd.DataFrame([cleaned_inputs], columns=MODEL_INPUT_FIELDS)
        new_patient_full = pd.DataFrame(0, index=[0], columns=ORIGINAL_COLUMNS)

        for column in new_patient.columns:
            if column in new_patient_full.columns:
                new_patient_full[column] = new_patient[column].values

        selected_input_frame = new_patient_full[SELECTED_FEATURES] if SELECTED_FEATURES else new_patient_full
        transformed_patient = SELECTOR.transform(new_patient_full)
        prediction = int(MODEL.predict(transformed_patient)[0])
        probabilities = MODEL.predict_proba(transformed_patient)[0]
        probability_map = dict(zip(MODEL.classes_, probabilities))
        transformed_list = transformed_patient[0].tolist()

        logger.info("new_patient row: %s", new_patient.to_dict(orient="records")[0])
        logger.info(
            "new_patient_full medical subset: %s",
            new_patient_full[MODEL_INPUT_FIELDS].to_dict(orient="records")[0],
        )
        logger.info(
            "selected feature values: %s",
            selected_input_frame.to_dict(orient="records")[0],
        )
        logger.info("transformed input: %s", transformed_list)
        logger.info("prediction: %s", prediction)
        logger.info(
            "probabilities_by_class: %s",
            {
                str(int(cls)): round(float(prob) * 100, 2)
                for cls, prob in zip(MODEL.classes_, probabilities)
            },
        )

        survived_probability = round(float(probability_map.get(0, 0.0) * 100), 2)
        death_probability = round(float(probability_map.get(1, 0.0) * 100), 2)

        response = {
            "result": "High Risk (Death)" if prediction == 1 else "Low Risk (Survived)",
            "survived_probability": survived_probability,
            "death_probability": death_probability,
        }

        if INCLUDE_PREDICTION_DEBUG:
            response["debug"] = {
                "received_input": medical_payload,
                "processed_input": cleaned_inputs,
                "new_patient": new_patient.to_dict(orient="records")[0],
                "new_patient_full_medical_subset": new_patient_full[
                    MODEL_INPUT_FIELDS
                ].to_dict(orient="records")[0],
                "selected_features": SELECTED_FEATURES,
                "selected_feature_values": selected_input_frame.to_dict(orient="records")[0],
                "missing_selected_features_from_ui": MISSING_SELECTED_FEATURES_FROM_UI,
                "model_classes": [int(cls) for cls in MODEL.classes_.tolist()],
                "first_transformed_input": transformed_list,
                "prediction": prediction,
                "probabilities_by_class": {
                    str(int(cls)): round(float(prob) * 100, 2)
                    for cls, prob in zip(MODEL.classes_, probabilities)
                },
                "y_value_counts": TRAINING_DEBUG_INFO.get("y_value_counts", {}),
                "startup_self_test": SELF_TEST_RESULT,
            }

        return jsonify(response)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        logger.exception("Prediction failed unexpectedly.")
        return jsonify({"error": "Prediction failed. Please try again."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
