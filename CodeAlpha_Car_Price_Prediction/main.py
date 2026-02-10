import os
import joblib

from src.preprocessing import load_data, preprocess_data
from src.model_trainer import train_and_select_best_model
from src.model_evaluation import evaluate_model


MODEL_DIR = "artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)


def run_pipeline():
    # 1. Load data
    df = load_data("Data/car data.csv")

    # 2. Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

    # 3. Train & select best model
    best_model, model_name, best_score = train_and_select_best_model(
        X_train, X_test, y_train, y_test
    )

    # 4. Evaluate model
    metrics = evaluate_model(best_model, X_test, y_test)

    # 5. Save artifacts
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.pkl"))

    print("\nPipeline Completed Successfully")
    print(f"Best Model : {model_name}")
    print(f"R2 Score   : {metrics['r2']:.3f}")


if __name__ == "__main__":
    run_pipeline()

