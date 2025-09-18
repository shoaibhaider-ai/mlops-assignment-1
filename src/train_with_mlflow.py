import os
import mlflow
import mlflow.sklearn
import joblib
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Configure MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("MLOPS")
X, y = load_wine(return_X_y=True)

# Load dataset
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True)
}

# Track best model
best_model = None
best_score = -1
best_name = ""

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        # Train
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="macro")
        rec = recall_score(y_test, preds, average="macro")
        f1 = f1_score(y_test, preds, average="macro")

        # Log parameters
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)

        # Save model artifact
        model_path = f"models/{name}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path, artifact_path="models")

        # Save and log confusion matrix
        disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
        plt.title(f"{name} Confusion Matrix")
        cm_path = f"results/{name}_cm.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")

        print(f"\n{name} logged to MLflow with Accuracy: {acc:.4f}")

        # Track best model by F1-score
        if f1 > best_score:
            best_score = f1
            best_model = model
            best_name = name

# 🔹 After loop: Register best model in MLflow registry
if best_model:
    print(f"\nBest Model: {best_name} with F1 = {best_score:.4f}")
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="best_model",
        registered_model_name="WineBestModel"
    )
