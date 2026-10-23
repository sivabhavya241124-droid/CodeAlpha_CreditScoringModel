from pathlib import Path
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

TARGET = "default payment next month"

def find_dataset():
    candidates = list(DATA_DIR.rglob("*.xls")) + list(DATA_DIR.rglob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError(
            "Dataset file not found. Put 'default of credit card clients.xls' inside the data folder."
        )
    return candidates[0]

def main():
    dataset_path = find_dataset()
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_excel(dataset_path, header=1)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found. Check the Excel file/header.")

    df = df.drop(columns=["ID"], errors="ignore")
    df = df.dropna(subset=[TARGET])

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42,
            class_weight="balanced", n_jobs=-1
        )
    }

    results = []
    fitted_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1-score": f1_score(y_test, predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probabilities)
        }
        results.append(row)
        fitted_models[name] = model
        print("  " + " | ".join(
            f"{metric}: {value:.4f}" for metric, value in row.items() if metric != "Model"
        ))

    results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print("\nModel comparison (sorted by F1-score):")
    print(results_df.to_string(index=False))

    # Select by F1-score so precision/recall balance is considered.
    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    joblib.dump(
        {"model": best_model, "features": list(X.columns), "model_name": best_name},
        MODEL_DIR / "best_model.pkl"
    )

    best_predictions = best_model.predict(X_test)
    best_probabilities = best_model.predict_proba(X_test)[:, 1]

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, best_predictions, ax=ax)
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, best_probabilities, ax=ax)
    ax.set_title(f"ROC Curve - {best_name}")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "roc_curve.png", dpi=150)
    plt.close(fig)

    print(f"\nBest model by F1-score: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'best_model.pkl'}")
    print(f"Saved reports in: {REPORT_DIR}")
    print("\nTraining complete. Next run: python -m streamlit run app.py")

if __name__ == "__main__":
    main()
