import os
import joblib
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load dataset
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True)
}

results = {}

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro")
    rec = recall_score(y_test, preds, average="macro")
    f1 = f1_score(y_test, preds, average="macro")

    results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    # Save model
    joblib.dump(model, f"models/{name}.pkl")

# Print results
print("Model Comparison:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

