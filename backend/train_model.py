from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Covid Data.csv"
MODEL_PATH = BASE_DIR / "covid_model.pkl"
SELECTOR_PATH = BASE_DIR / "selector.pkl"
COLUMNS_PATH = BASE_DIR / "columns.pkl"


def train_and_save_model():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{DATA_PATH}'. Please add 'Covid Data.csv' first."
        )

    df = pd.read_csv(DATA_PATH)

    df["DEATH"] = df["DATE_DIED"].apply(lambda x: 0 if x == "9999-99-99" else 1)
    df = df.drop(columns=["DATE_DIED"])

    df = df.replace(2, 0)

    cols_to_clean = df.columns.drop("AGE")
    df[cols_to_clean] = df[cols_to_clean].replace([97, 98, 99], 0)

    X = df.drop(columns=["DEATH"])
    y = df["DEATH"]

    selector = SelectKBest(score_func=chi2, k=10)
    X_reduced = selector.fit_transform(X, y)

    selected_features = X.columns[selector.get_support()]

    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42
    )

    dt_model = DecisionTreeClassifier(
        max_depth=10,
        class_weight={0: 1, 1: 3},
        random_state=42,
    )

    dt_model.fit(X_train, y_train)

    y_pred = dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(dt_model, MODEL_PATH)
    joblib.dump(selector, SELECTOR_PATH)
    joblib.dump(list(X.columns), COLUMNS_PATH)

    print("Training complete.")
    print(f"Selected features: {', '.join(selected_features)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved selector to: {SELECTOR_PATH}")
    print(f"Saved columns to: {COLUMNS_PATH}")


if __name__ == "__main__":
    train_and_save_model()
