# Healthcare ML Project

End-to-end healthcare analytics system: data ingestion, cleaning, PostgreSQL storage, weekly ML retraining, and a live FastAPI prediction API.


## Project Overview

This system ingests a synthetic healthcare dataset (10 000 records from Kaggle), cleans and stores it in PostgreSQL, trains a multi-class classifier to predict patient test results (**Normal / Abnormal / Inconclusive**), and exposes predictions through a REST API with a simple HTML frontend.

### Architecture

```
healthcare-ml-project/
├── app/              # FastAPI application
│   ├── main.py       # App factory + APScheduler (Saturday 12:00 UTC retrain)
│   ├── routes.py     # Endpoint handlers
│   ├── schemas.py    # Pydantic request / response models
│   ├── model_loader.py
│   └── utils.py
├── data/
│   └── raw/          # Place healthcare_dataset.csv here
├── database/
│   ├── db_connection.py   # SQLAlchemy engine + session context manager
│   ├── models.py          # ORM table definitions
│   └── queries.sql        # Analytical SQL snippets
├── ml/
│   ├── preprocess.py  # Feature pipeline (StandardScaler + OneHotEncoder)
│   ├── train.py       # Multi-model training (XGBoost, RF, LR)
│   ├── evaluate.py    # Metrics (accuracy, precision, recall, F1, CM)
│   └── predict.py     # Inference with cached model artefact
├── models/            # Saved .joblib artefacts (git-ignored)
├── notebooks/         # Jupyter EDA notebook
├── scripts/
│   ├── ingest.py      # Load raw CSV → raw_patients table
│   ├── clean.py       # Clean raw → cleaned_patients table
│   └── load.py        # Verify / summarise loaded data
├── frontend/
│   └── index.html     # Standalone prediction UI
└── tests/
    ├── test_api.py
    └── test_ml.py
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.11 |
| UV (package manager) | latest |
| PostgreSQL | ≥ 14 |

Install UV:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/healthcare-ml-project.git
cd healthcare-ml-project
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

```bash
cp .env
# Edit .env with your PostgreSQL credentials
```

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=healthcare_db
DB_USER=postgres
DB_PASSWORD=your_password
```

### 4. Create the PostgreSQL database

```sql
-- Run in psql
CREATE DATABASE healthcare_db;
```

### 5. Download the dataset

Option A — Kaggle Hub script:
```bash
uv run python -c "
import kagglehub, shutil, pathlib
p = kagglehub.dataset_download('prasad22/healthcare-dataset')
dest = pathlib.Path('data/raw/healthcare_dataset.csv')
dest.parent.mkdir(parents=True, exist_ok=True)
next(pathlib.Path(p).glob('*.csv'))  # find the CSV
shutil.copy(next(pathlib.Path(p).glob('*.csv')), dest)
print('Saved to', dest)
"
```

Option B — Download manually from  
<https://www.kaggle.com/datasets/prasad22/healthcare-dataset/data>  
and place the file at `data/raw/healthcare_dataset.csv`.

### 6. Ingest, clean, and verify the data

```bash
# Load raw CSV into PostgreSQL
uv run python scripts/ingest.py

# Clean and transform into training-ready table
uv run python scripts/clean.py

# Verify record counts and class distribution
uv run python scripts/load.py
```

### 7. Train the model

```bash
uv run python ml/train.py
```

This trains XGBoost, Random Forest, and Logistic Regression, selects the best by weighted F1, and saves it to `models/model.joblib`.

---

## Running the API

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

| URL | Description |
|-----|-------------|
| <http://localhost:8000/docs> | Interactive Swagger UI |
| <http://localhost:8000/health> | Health / readiness check |
| <http://localhost:8000/predict> | POST — run prediction |
| <http://localhost:8000/train> | POST — trigger retraining |
| <http://localhost:8000/model-info> | GET — loaded model metadata |
| <http://localhost:8000/ui> | HTML prediction frontend |

---

## API Reference

### POST `/predict`

**Request body**

```json
{
  "Age": 45,
  "Gender": "Male",
  "Blood Type": "O+",
  "Medical Condition": "Diabetes",
  "Billing Amount": 2000.50,
  "Admission Type": "Emergency",
  "Insurance Provider": "Cigna",
  "Medication": "Aspirin"
}
```

**Response**

```json
{
  "predicted_test_result": "Abnormal",
  "model_name": "XGBoost"
}
```

**cURL example**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 45,
    "Gender": "Male",
    "Blood Type": "O+",
    "Medical Condition": "Diabetes",
    "Billing Amount": 2000.50,
    "Admission Type": "Emergency",
    "Insurance Provider": "Cigna",
    "Medication": "Aspirin"
  }'
```

### GET `/health`

```json
{
  "status": "ok",
  "db_connected": true,
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

---

## Scheduled Retraining

The APScheduler job inside `app/main.py` fires **every Saturday at 12:00 noon UTC**.  
It reads the latest cleaned data from PostgreSQL, retrains all three models, picks the best, saves it, and reloads the in-memory model — with zero downtime.

To trigger a manual retrain:

```bash
curl -X POST http://localhost:8000/train
# → HTTP 202 Accepted (runs in background)
```

---

## Deployment (Render.com — free tier)

1. Push the project to a public GitHub repository.
2. Go to <https://render.com> → **New Web Service**.
3. Connect your GitHub repo.
4. Set build command: `pip install uv && uv sync`
5. Set start command: `feat: Create static UI for healthcare prediction form

- Add HTML form with patient attribute input fields
- Integrate Swagger/OpenAPI documentation links
- Include styling and form validation on client side
- Connect to FastAPI backend via /predict endpoint`
6. Add environment variables (DB credentials, etc.) in the Render dashboard.
7. Click **Deploy**.

A `render.yaml` is included for one-click Infrastructure-as-Code deployment.

---

## ML Models & Evaluation

| Model | Notes |
|-------|-------|
| **XGBoost** | Gradient boosting — typically best performer |
| Random Forest | Ensemble baseline |
| Logistic Regression | Linear baseline |

Evaluation metrics reported per run:
- Accuracy, Precision (weighted), Recall (weighted), F1 (weighted)
- Confusion matrix
- Full per-class classification report

All metrics are persisted to the `model_runs` PostgreSQL table for auditing.

---
