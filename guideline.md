# Cloud-Native Machine Learning Pipeline using BigQuery, Docker, and Kubernetes

This practical project demonstrates a complete cloud-native machine learning workflow using real-world COVID-19 data from BigQuery. The pipeline covers data collection, preprocessing, feature engineering, machine learning, dashboard deployment, Docker containerization, and Kubernetes orchestration.

The goal is to expose learners to modern AI deployment workflows used in real production systems.

---

# Overall Pipeline

```text id="7jlwma"
BigQuery Dataset
        ↓
Data Collection
        ↓
Preprocessing & Feature Engineering
        ↓
Machine Learning Models
        ↓
Interactive Streamlit Dashboard
        ↓
Docker Containerization
        ↓
Kubernetes Orchestration
```

---

# System Architecture

```text id="6jlwmb"
Google BigQuery
        ↓
Python Data Pipeline
        ↓
Processed Dataset
        ↓
Regression + Classification Models
        ↓
Streamlit Dashboard
        ↓
Docker Image
        ↓
Kubernetes Pods & Services
```

---

# Project Folder Layout

```text id="5jlwmc"
week5_cloud_fog/

│
├── api/
│   └── app.py
│
├── credentials/
│   └── credentials.json
│
├── data/
│   ├── raw/
│   │   └── uganda_covid_data.csv
│   │
│   └── processed/
│       └── uganda_covid_processed.csv
│
├── models/
│   ├── covid_regression_model.pkl
│   └── covid_classification_model.pkl
│
├── scripts/
│   ├── bigquery_test.py
│   ├── preprocess.py
│   └── train_models.py
│
├── dashboard.py
├── Dockerfile
├── deployment.yaml
├── requirements.txt
│
└── README.md
```

---

# Stage 1 — BigQuery Data Collection

The project begins by collecting real-world COVID-19 data from Google BigQuery public datasets.

Core tasks:

* Authenticate using Google Cloud credentials
* Connect to BigQuery
* Query Uganda COVID-19 records
* Save dataset locally as CSV

Main technologies:

* Python
* Google BigQuery
* Pandas

Documentation:

* [Google BigQuery Documentation](https://cloud.google.com/bigquery/docs?utm_source=chatgpt.com)
* [Google Cloud Python SDK](https://cloud.google.com/python/docs/reference?utm_source=chatgpt.com)
* [Pandas Documentation](https://pandas.pydata.org/docs/?utm_source=chatgpt.com)

---

# Stage 2 — Data Preprocessing & Feature Engineering

The raw dataset is cleaned and transformed into machine-learning-ready features.

Key preprocessing operations:

* Missing value handling
* Date conversion
* Rolling averages
* Growth rate calculation
* Feature selection
* Risk interval categorization

Generated features:

* previous_day_cases
* rolling_avg_cases
* growth_rate
* risk_level

Machine learning problem types:

1. Regression → Predict future cases
2. Classification → Predict outbreak severity category

Main technologies:

* Pandas
* NumPy
* Matplotlib
* Scikit-learn

Documentation:

* [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html?utm_source=chatgpt.com)
* [Matplotlib Documentation](https://matplotlib.org/stable/index.html?utm_source=chatgpt.com)
* [NumPy Documentation](https://numpy.org/doc/?utm_source=chatgpt.com)

---

# Stage 3 — Machine Learning Model Training

Two machine learning models are trained:

## Regression Model

Predicts future COVID-19 case counts.

## Classification Model

Predicts outbreak severity levels:

* Low
* Moderate
* High
* Critical

Algorithms explored:

* Linear Regression
* Random Forest
* Logistic Regression
* Decision Trees

Evaluation metrics:

* Mean Squared Error (MSE)
* R² Score
* Accuracy
* Classification Report
* Confusion Matrix

Output:

* Serialized `.pkl` model files

Documentation:

* [Scikit-learn Model Persistence](https://scikit-learn.org/stable/model_persistence.html?utm_source=chatgpt.com)
* [Joblib Documentation](https://joblib.readthedocs.io/en/latest/?utm_source=chatgpt.com)

---

# Stage 4 — Streamlit Dashboard

An interactive dashboard is created to:

* accept user inputs,
* make predictions,
* visualize trends,
* display outbreak risk categories.

Dashboard capabilities:

* Real-time predictions
* Interactive charts
* Dataset preview
* Risk visualization

Main technologies:

* Streamlit
* Matplotlib

Documentation:

* [Streamlit Documentation](https://docs.streamlit.io/?utm_source=chatgpt.com)
* [Streamlit Components Guide](https://docs.streamlit.io/library/components?utm_source=chatgpt.com)

---

# Stage 5 — Docker Containerization

The Streamlit application is containerized using Docker.

Purpose:

* Package the application and dependencies
* Ensure portability
* Simplify deployment

Core Docker concepts:

* Dockerfile
* Image
* Container
* Port mapping
* Build context

Key commands:

Build image:

```bash id="4jlwmd"
docker build -t covid-dashboard .
```

Run container:

```bash id="3jlwme"
docker run -p 8501:8501 covid-dashboard
```

Documentation:

* [Docker Documentation](https://docs.docker.com/?utm_source=chatgpt.com)
* [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/?utm_source=chatgpt.com)
* [Docker Desktop Documentation](https://docs.docker.com/desktop/?utm_source=chatgpt.com)

---

# Stage 6 — Kubernetes Orchestration

The Dockerized application is deployed into Kubernetes.

Core Kubernetes components:

* Pods
* Deployments
* Services
* Replicas
* Port Forwarding

Kubernetes capabilities demonstrated:

* Container orchestration
* Automatic scaling
* Fault tolerance
* Deployment management

Deployment workflow:

Apply deployment:

```bash id="2jlwmf"
kubectl apply -f deployment.yaml
```

View pods:

```bash id="1jlwmg"
kubectl get pods
```

Scale deployment:

```bash id="0jlwmh"
kubectl scale deployment covid-dashboard-deployment --replicas=4
```

Access application:

```bash id="9jlwmi"
kubectl port-forward service/covid-dashboard-service 8501:8501
```

Documentation:

* [Kubernetes Documentation](https://kubernetes.io/docs/home/?utm_source=chatgpt.com)
* [kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/?utm_source=chatgpt.com)
* [Kubernetes Deployments Guide](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/?utm_source=chatgpt.com)

---

# Dockerfile Structure

```dockerfile id="8jlwmj"
FROM python:3.11

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

# Kubernetes Deployment Structure

Main Kubernetes resources:

* Deployment
* Service
* Replicas
* Pod networking

Key concepts demonstrated:

* Container orchestration
* Desired state management
* Service exposure
* Scaling applications

---

# Educational Outcomes

By completing this pipeline, learners gain practical experience in:

## Cloud Computing

* BigQuery
* Cloud datasets
* Data pipelines

## Machine Learning

* Regression
* Classification
* Feature engineering
* Model evaluation

## MLOps

* Model serialization
* Deployment pipelines
* Production environments

## Docker

* Containerization
* Image management
* Environment portability

## Kubernetes

* Orchestration
* Scaling
* Service management
* Pod lifecycle

---

# Suggested Extensions

Learners may further extend the project by:

* deploying to Google Kubernetes Engine (GKE),
* integrating CI/CD pipelines,
* adding monitoring dashboards,
* implementing REST APIs,
* using real-time streaming data,
* adding authentication systems,
* integrating cloud storage.

---

# Recommended Advanced Reading

## Cloud & Data Engineering

* [Google Cloud Skills Boost](https://www.cloudskillsboost.google/?utm_source=chatgpt.com)

## Machine Learning Operations (MLOps)

* [MLOps Guide by Google Cloud](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?utm_source=chatgpt.com)

## Docker & Containers

* [Docker Getting Started Guide](https://docs.docker.com/get-started/?utm_source=chatgpt.com)

## Kubernetes

* [Kubernetes Interactive Tutorials](https://kubernetes.io/docs/tutorials/?utm_source=chatgpt.com)

## Streamlit

* [Streamlit Gallery](https://streamlit.io/gallery?utm_source=chatgpt.com)
