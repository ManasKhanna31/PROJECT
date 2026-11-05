Program 1


# Install required libraries ! or %
!pip install -q numpy pandas scikit-learn joblib mlflow flask pytest pyngrok
# Configure and initialize Git
!git config --global user.email "student@example.com"
!git config --global user.name "MLOps Student"
!git init -q
!git add .
!git commit -m "Initial commit" -q || echo "Nothing to commit yet"
!git log --oneline

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
iris=load_iris()
X=pd.DataFrame(iris.data,columns=iris.feature_names)
y=pd.Series(iris.target)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=RandomForestClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)

Program 2


import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))

joblib.dump(model, 'diabetes_model.joblib')

loaded_model = joblib.load('diabetes_model.joblib')
loaded_pred = loaded_model.predict(X_test)

print("Loaded MSE:", mean_squared_error(y_test, loaded_pred))

----
%%writefile app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('diabetes_model.joblib')

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': float(prediction[0])})
----

import threading
import time
import requests

thread = threading.Thread(
    target=lambda: app.run(port=5054, use_reloader=False)
)
----

from app import app

thread = threading.Thread(
    target=lambda: app.run(port=5054, use_reloader=False)
)
thread.daemon = True
thread.start()

time.sleep(2)

sample = X_test.iloc[0].tolist()

response = requests.post(
    'http://127.0.0.1:5054/predict',
    json={'features': sample}
)

print(response.json())

Program 3

%pip install -q pandas numpy scikit-learn joblib pytest
import pandas as pd
import joblib

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
data = load_diabetes()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# Save model
joblib.dump(model, "diabetes_model.joblib")

print("Model trained successfully!")
print("MSE:", mse)
print("Model saved successfully!")
----
%%writefile test_model.py

import joblib
import numpy as np

def test_model_prediction():
    model = joblib.load("diabetes_model.joblib")

    input_data = np.array([
        0.02, 0.07, 0.03, 0.02, 0.01,
        0.03, 0.01, 0.04, 0.03, 0.01
    ]).reshape(1, -1)

    prediction = model.predict(input_data)

    assert isinstance(prediction[0], (float, np.floating)), \
        f"Expected float but got {type(prediction[0])}"

def test_model_output_shape():
    model = joblib.load("diabetes_model.joblib")

    input_data = np.zeros((5, 10))
    predictions = model.predict(input_data)

    assert predictions.shape == (5,), \
        f"Expected shape (5,) but got {predictions.shape}"
----
!pytest test_model.py -v
----
import os
os.makedirs(".github/workflows",exist_ok=True)
----
%%writefile ci.yml

name: MLOps CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install numpy pandas scikit-learn joblib pytest

      - name: Train model
        run: python train.py

      - name: Run unit tests
        run: pytest test_model.py -v

  deploy:
    needs: build-and-test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Deploy model
        run: |
          echo "Deploying trained model..."
          # gcloud run deploy mlops-api --source . --region us-central1

Program 4


import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
data = load_diabetes()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set experiment
mlflow.set_experiment("diabetes-regression")

# Models to compare
models_to_try = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

# Train and track models
for name, mdl in models_to_try.items():

    with mlflow.start_run(run_name=name):

        mdl.fit(X_train, y_train)

        y_pred = mdl.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        # Log parameters
        mlflow.log_param("model_type", name)

        if name == "Ridge":
            mlflow.log_param("alpha", 1.0)

        if name == "RandomForest":
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)

        # Log metrics
        mlflow.log_metric("mse", mse)

        # Log model artifact using pickle
        mlflow.sklearn.log_model(
            mdl,
            name="model",
            serialization_format="pickle"
        )

        print(f"{name}: MSE = {mse:.2f}")
----
import subprocess
import time

mlflow_process = subprocess.Popen(
    ["mlflow", "ui", "--port", "5001"]
)

time.sleep(5)

print("MLflow UI started!")
----
from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN")

public_url = ngrok.connect(5001)

print("MLflow UI:", public_url)

Program 5

%pip install prometheus_client
----
# Install required library
%pip install -q prometheus_client

import logging
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from prometheus_client import start_http_server, Gauge, REGISTRY
from IPython.display import IFrame, display

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mlops_monitoring.log')
    ],
    force=True
)

# Load dataset
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()

logging.info("training started")
model.fit(X_train, y_train)
logging.info("training completed")

# Evaluate model
mse = mean_squared_error(y_test, model.predict(X_test))

# Measure prediction latency
start = time.time()
model.predict(X_test)
latency = time.time() - start

logging.info(f"MSE: {mse}")

# Performance alert
if mse > 3500:
    logging.warning("ALERT: Model performance degraded")
else:
    logging.info("normal")

print("mse:", mse)
print("latency:", latency)

# Remove previous custom metrics if rerunning
for name in ['model_mse', 'prediction_latency_seconds']:
    if name in REGISTRY._names_to_collectors:
        REGISTRY.unregister(REGISTRY._names_to_collectors[name])

# Start Prometheus metrics server
start_http_server(8000)

# Create metrics
MSE_GAUGE = Gauge('model_mse', 'Model MSE')
LATENCY_GAUGE = Gauge(
    'prediction_latency_seconds', 'Prediction Latency'
)

# Set metrics
MSE_GAUGE.set(mse)
LATENCY_GAUGE.set(latency)

print("Metrics available at http://127.0.0.1:8000/metrics")

# Display metrics inside Jupyter Notebook
display(IFrame(
    src="http://127.0.0.1:8000/metrics",
    width=900,
    height=400
))



https://github.com/ManasKhanna31/PROJECT.git