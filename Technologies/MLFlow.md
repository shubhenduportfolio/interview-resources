# ✅ Chapter 1: Introduction to MLflow
## 🎯 What is MLflow?
MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It helps you keep track of experiments, package ML code, reproduce runs, share models, and deploy them—all in one place.

It supports any ML library (scikit-learn, PyTorch, TensorFlow, XGBoost, etc.) and is language-agnostic, although it has strong Python support.

## 🚀 Why Use MLflow in ML Projects?
### Without MLflow:
- Your experiments are scattered across notebooks or folders.
- Difficult to reproduce which model was trained with what parameters.
- Manual tracking of metrics in spreadsheets.
- Deployment involves custom scripts.

### With MLflow:
- Track parameters, metrics, models, and outputs automatically.
- Compare runs visually via a web UI.
- Package and share your code in reusable format.
- Easily deploy or serve models via API.
- Centralized Model Registry to manage multiple versions.

MLflow simplifies collaboration, reproducibility, and production-readiness in ML workflows.

## 🧩 Overview of MLflow Components
MLflow consists of four major components:

| Component | Description |
|---|---|
| MLflow Tracking | Record and query experiments: parameters, metrics, artifacts, and logs. |
| MLflow Projects | Package your code with environment details for reproducibility. |
| MLflow Models | Format to save/load ML models with multiple deployment options. |
| Model Registry | Central store to manage models, versioning, stage transitions, approvals. | 

We'll go deeper into each component in later chapters.

## 🛠️ Installation and Setup
You can install MLflow via pip:

```
pip install mlflow
```
To verify installation and check the version:

```
mlflow --version
```
MLflow also runs a local server for experiment tracking. Start it using:

```
mlflow ui
```
This starts a web server on http://localhost:5000 by default.

✅ First Hello World Experiment with MLflow
Let’s now do a simple machine learning experiment using scikit-learn and track everything with MLflow.

📂 Folder Structure
```
mlflow_hello_world/
│
├── train.py

```
## 🧪 Step-by-Step: Code Explanation
Create a file named train.py with the following code:

```
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Enable automatic logging (optional)
mlflow.sklearn.autolog()

# Start an MLflow run
with mlflow.start_run():

    # Define model and train
    model = RandomForestClassifier(n_estimators=100, max_depth=3)
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    # Manually log metrics (optional if autolog is used)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 3)
    mlflow.log_metric("accuracy", acc)

    # Log model artifact
    mlflow.sklearn.log_model(model, "model")

    print("Logged to MLflow")


```
## ▶️ Run the Script
```
python train.py
```
This will:

- Create an experiment
- Log the run
- Save metrics, parameters, and the trained model

## 🔎 Visualize in the UI
Start the MLflow tracking UI:

```
mlflow ui
```
Visit: http://localhost:5000

You’ll see:

- Each run listed
- Parameters like n_estimators, max_depth
- Metrics like accuracy
- Artifacts (model, etc.)
- Source code snapshot

## 📦 Artifacts and Files
After a run, MLflow stores metadata and model files in a default directory:
```
mlruns/
└── 0/
    └── <run_id>/
        ├── metrics/
        ├── params/
        ├── artifacts/
        └── meta.yaml
```
# 🧠 Key Concepts Recap
- Term	Meaning
- Run	A single experiment execution
- Experiment	Group of related runs
- Artifact	Output files like models or plots
- Tracking URI	The destination (local/remote) where data is logged
