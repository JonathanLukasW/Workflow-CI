import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth",    type=int, default=10)
    args = parser.parse_args()

    TRAIN_PATH = os.path.join("csgo_preprocessed", "train_clean.csv")
    TEST_PATH  = os.path.join("csgo_preprocessed", "test_clean.csv")

    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Membaca dataset re-training...")
    train_data = pd.read_csv(TRAIN_PATH)
    test_data  = pd.read_csv(TEST_PATH)

    X_train = train_data.drop(columns=["winningSide"])
    y_train = train_data["winningSide"]
    X_test  = test_data.drop(columns=["winningSide"])
    y_test  = test_data["winningSide"]

    print(f"  Train shape : {X_train.shape}")
    print(f"  Test  shape : {X_test.shape}")

    mlflow.autolog(log_model_signatures=True, log_input_examples=False)

    print(f"\nMemulai Retraining Model (n_estimators={args.n_estimators}, max_depth={args.max_depth})")
    with mlflow.start_run(run_name="CI_Automated_Retraining"):

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        print("Proses training berhasil!")

        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_metric("test_accuracy", acc)
        print(f"  Test Accuracy : {acc:.4f}")

        report_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact(report_path, artifact_path="outputs")
        print(f"  Saved : {report_path}")

        importances = model.feature_importances_
        feat_names  = X_train.columns.tolist()
        feat_df = (
            pd.DataFrame({"feature": feat_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(
            feat_df["feature"][::-1],
            feat_df["importance"][::-1],
            color="#4E9AF1",
            edgecolor="white",
        )
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_title("Top-10 Feature Importances (Random Forest)", fontweight="bold", pad=12)
        ax.set_xlabel("Importance Score")
        ax.tick_params(left=False)
        plt.tight_layout()

        fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
        fig.savefig(fi_path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(fi_path, artifact_path="outputs")
        print(f"  Saved : {fi_path}")

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(y_test))
        ax2.scatter(
            y_test + jitter,
            y_pred + jitter,
            alpha=0.25,
            s=8,
            color="#F47C3C",
            label="Prediksi",
        )
        ax2.plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 "k--", linewidth=1.2, label="Perfect prediction")
        ax2.set_title("Residual Plot – Predicted vs Actual", fontweight="bold", pad=12)
        ax2.set_xlabel("Actual Label")
        ax2.set_ylabel("Predicted Label")
        ax2.set_xticks([0, 1]); ax2.set_xticklabels(["CT (0)", "T (1)"])
        ax2.set_yticks([0, 1]); ax2.set_yticklabels(["CT (0)", "T (1)"])
        ax2.legend()
        plt.tight_layout()

        rp_path = os.path.join(OUTPUT_DIR, "residual_plot.png")
        fig2.savefig(rp_path, dpi=120)
        plt.close(fig2)
        mlflow.log_artifact(rp_path, artifact_path="outputs")
        print(f"  Saved : {rp_path}")

        mlflow.sklearn.log_model(model, "random_forest_model")
        print("\nSemua artifact berhasil di-log ke MLflow!")
        print(f"Folder outputs/ berisi : {os.listdir(OUTPUT_DIR)}")