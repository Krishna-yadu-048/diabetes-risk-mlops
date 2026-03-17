# 🩺 Diabetes Risk Predictor — MLOps Portfolio Project

A end-to-end MLOps project that predicts diabetes risk from clinical measurements.
Three models (Logistic Regression, Random Forest, XGBoost) are trained, tracked,
and compared in MLflow. The best model is served via a FastAPI REST API and an
interactive prediction dashboard.

Built using the **MLOps V2** template: lean, completable, and deployable without
expensive cloud infrastructure.

---

## 📸 Dashboard Preview

> After running `make serve`, open [http://localhost:8000](http://localhost:8000)

The dashboard lets anyone enter patient vitals and get a diabetes risk prediction
with a confidence score — no API knowledge required.

---

## 🛠️ Tech Stack

| Tool | Layer | Purpose |
|---|---|---|
| **DVC** | Data Versioning | Versions `data/` outside of Git. Works locally or with DagsHub. |
| **MLflow** | Experiment Tracking | Logs all 3 model runs. Registry promotes best model to Production. |
| **FastAPI** | Model Serving + Dashboard | REST API at `/predict` and HTML dashboard at `/`. |
| **Docker** | Environment | Packages the app into a container for consistent deployment. |
| **GitHub Actions** | CI / CD | Runs tests + linting on push. Builds Docker image on merge to main. |
| **Pytest** | Testing | Unit tests for API endpoints and ML logic. MLflow is mocked. |

---

## 📁 Project Structure

```
diabetes-risk-mlops/
│
├── .github/workflows/
│   ├── ci_pipeline.yml       # Lint + test on every push
│   └── cd_pipeline.yml       # Build + push Docker image on merge to main
│
├── data/
│   ├── raw/                  # Original CSV (tracked by DVC, ignored by Git)
│   └── processed/            # Cleaned output from validate.py
│
├── notebooks/
│   └── README.md             # Placeholder for EDA notebooks
│
├── src/
│   ├── validate.py           # Hand-written data validation + cleaning
│   ├── train.py              # Trains LR, RF, XGBoost — logs all to MLflow
│   └── evaluate.py           # Evaluates Production model on test set
│
├── api/
│   ├── main.py               # FastAPI app — dashboard + REST endpoints
│   ├── schemas.py            # Pydantic request/response models
│   ├── Dockerfile            # Multi-stage Docker build
│   ├── static/style.css      # Dashboard styling
│   └── templates/index.html  # Jinja2 prediction dashboard
│
├── tests/
│   ├── test_api.py           # FastAPI endpoint tests (MLflow mocked)
│   └── test_model.py         # validate(), clean(), and sklearn model tests
│
├── dvc.yaml                  # DVC pipeline: validate → train → evaluate
├── pyproject.toml            # All dependencies (core + dev + dagshub)
├── Makefile                  # Shortcuts for every common command
├── .env.example              # Environment variable template
└── README.md
```

---

## 🚀 Quickstart (Local)

### 1. Clone and install

**Windows (PowerShell):**
```powershell
git clone https://github.com/<your-username>/diabetes-risk-mlops.git
cd diabetes-risk-mlops

# Allow PowerShell to run local scripts (one-time setup)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

.\run.ps1 setup
```

**macOS / Linux:**
```bash
git clone https://github.com/<your-username>/diabetes-risk-mlops.git
cd diabetes-risk-mlops
make setup
```

This creates a `.venv/` virtual environment and installs all dependencies into it.

### 2. Get the dataset

Download the **Pima Indians Diabetes Dataset** from Kaggle:
[https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

Place the file at:
```
data/raw/diabetes.csv
```

### 3. Initialise DVC and version the data

**Windows:**
```powershell
.\run.ps1 dvc-init
.\run.ps1 dvc-push
```
**macOS / Linux:**
```bash
make dvc-init
```

### 4. Run the full pipeline

**Windows:**
```powershell
.\run.ps1 validate   # clean and validate the raw data
.\run.ps1 train      # train all 3 models, log to MLflow
                      # Production alias is set automatically on the best model
```
**macOS / Linux:**
```bash
make validate
make train
```

### 5. Evaluate and serve

**Windows:**
```powershell
.\run.ps1 evaluate   # evaluate the Production model
.\run.ps1 serve      # start FastAPI at http://localhost:8000
```
**macOS / Linux:**
```bash
make evaluate
make serve
```

Visit [http://localhost:8000](http://localhost:8000) for the dashboard,
or [http://localhost:8000/docs](http://localhost:8000/docs) for the Swagger UI.

### 6. View MLflow experiment runs (optional)

**Windows:**
```powershell
.\run.ps1 mlflow-ui
```
**macOS / Linux:**
```bash
make mlflow-ui
```
Open **http://127.0.0.1:5000** → Experiments → diabetes-risk

---

## 🧪 Running Tests

```bash
make test
```

Tests run without a live MLflow server — the model is mocked via `unittest.mock`.

```bash
make lint       # check for issues
make lint-fix   # auto-fix where possible
```

---

## 🌿 Optional: Connect DagsHub

DagsHub gives you a **hosted MLflow UI** with a public URL — useful for showing
recruiters your experiment runs without them having to run anything locally.

### Setup

1. Create a free account at [https://dagshub.com](https://dagshub.com)
2. Create a new repo and push your project to it
3. On DagsHub, go to your repo → **Remote → Experiments** to get your tracking URI
4. Copy `.env.example` to `.env` and fill in:

```bash
MLFLOW_TRACKING_URI=https://dagshub.com/<your_username>/<your_repo>.mlflow
MLFLOW_TRACKING_USERNAME=<your_dagshub_username>
MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>
```

5. Install the optional DagsHub dependency:

```bash
pip install -e ".[dagshub]"
```

6. Run training as normal — all runs will now appear in your DagsHub experiment page:

```bash
make train
```

> **Note:** The project works 100% locally without DagsHub.
> DagsHub is a bonus layer for sharing your experiment results publicly.

---

## 🐳 Docker

Build and run the API in a container:

```bash
# Build
docker build -t diabetes-risk-mlops -f api/Dockerfile .

# Run (mount local mlruns so the container can find the Production model)
docker run -p 8000:8000 \
  -e MODEL_NAME=diabetes-risk-model \
  -e MLFLOW_TRACKING_URI=mlruns \
  -v $(pwd)/mlruns:/app/mlruns \
  diabetes-risk-mlops
```

---

## 🔁 CI / CD Pipeline

| Trigger | Action |
|---|---|
| Push to any branch | Lint with `ruff` + run all `pytest` tests |
| Merge to `main` | Build Docker image + push to Docker Hub |

To enable CD, add these secrets to your GitHub repo
(**Settings → Secrets → Actions**):

| Secret | Value |
|---|---|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Your Docker Hub access token |

Also update the image tag in `.github/workflows/cd_pipeline.yml`:
```yaml
tags: yourusername/diabetes-risk-mlops:latest
       # ↑ replace with your Docker Hub username
```

---

## 📊 API Endpoints

| Route | Method | Description |
|---|---|---|
| `/` | GET | Prediction dashboard — form UI for non-technical users |
| `/dashboard-predict` | POST | Handles dashboard form submission |
| `/predict` | POST | REST endpoint — JSON in, JSON out |
| `/health` | GET | Model load status |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc API docs |

### Example `/predict` request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 2,
    "Glucose": 138,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 80,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 47
  }'
```

### Example response

```json
{
  "prediction": 1,
  "prediction_label": "Diabetes Risk Detected",
  "confidence": 0.82,
  "model_name": "diabetes-risk-model"
}
```

---

## 📋 Commands

All commands are available via `run.ps1` on Windows or `make` on macOS/Linux.

| Command | Windows | macOS / Linux |
|---|---|---|
| Install dependencies | `.\run.ps1 setup` | `make setup` |
| Initialise DVC | `.\run.ps1 dvc-init` | `make dvc-init` |
| Push data to DVC remote | `.\run.ps1 dvc-push` | `make dvc-push` |
| Pull data from DVC remote | `.\run.ps1 dvc-pull` | `make dvc-pull` |
| Validate and clean data | `.\run.ps1 validate` | `make validate` |
| Train all 3 models | `.\run.ps1 train` | `make train` |
| Evaluate Production model | `.\run.ps1 evaluate` | `make evaluate` |
| Run full DVC pipeline | `.\run.ps1 pipeline` | `make pipeline` |
| Start FastAPI server | `.\run.ps1 serve` | `make serve` |
| Open MLflow UI | `.\run.ps1 mlflow-ui` | `make mlflow-ui` |
| Run all tests | `.\run.ps1 test` | `make test` |
| Run linter | `.\run.ps1 lint` | `make lint` |
| Auto-fix lint issues | `.\run.ps1 lint-fix` | `make lint-fix` |
| Clean generated files | `.\run.ps1 clean` | `make clean` |
| Remove .venv | `.\run.ps1 clean-venv` | `make clean-venv` |

---

## 📦 Dataset

**Pima Indians Diabetes Database**
- Source: UCI Machine Learning Repository / Kaggle
- 768 rows, 8 clinical features, 1 binary target (`Outcome`)
- Notable: Zero values in physiological columns encode missing data —
  `validate.py` replaces these with NaN and drops the affected rows

---

## ⚠️ Disclaimer

This project is for educational and portfolio purposes only.
It is not a medical tool and should not be used for clinical decision-making.
