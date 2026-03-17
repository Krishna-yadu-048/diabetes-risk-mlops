# 🩺 Diabetes Risk Predictor — MLOps Portfolio Project

An end-to-end MLOps project that predicts diabetes risk from clinical measurements.
Three models (Logistic Regression, Random Forest, XGBoost) are trained, tracked,
and compared in MLflow. The best model is automatically promoted and served via a
FastAPI REST API and an interactive prediction dashboard.

Built using the **MLOps V2** template: lean, completable, and deployable without
expensive cloud infrastructure.

---

## 📸 Dashboard Preview

After running `.\run.ps1 serve` (Windows) or `make serve` (macOS/Linux),
open [http://localhost:8000](http://localhost:8000)

The dashboard lets anyone enter patient vitals and get a diabetes risk prediction
with a confidence score — no API knowledge required.

---

## 🛠️ Tech Stack

| Tool | Layer | Purpose |
|---|---|---|
| **DVC** | Data Versioning | Versions `data/` outside of Git. Works locally or with DagsHub. |
| **MLflow** | Experiment Tracking | Logs all 3 model runs. Best model is auto-promoted via alias. |
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
├── run.ps1                   # Windows PowerShell command runner
├── Makefile                  # macOS / Linux command runner
├── dvc.yaml                  # DVC pipeline: validate → train → evaluate
├── pyproject.toml            # All dependencies (core + dev + dagshub)
├── .env.example              # Environment variable template
└── README.md
```

---

## 🚀 Quickstart (Local)

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/diabetes-risk-mlops.git
cd diabetes-risk-mlops
```

### 2. Install dependencies

**Windows (PowerShell):**
```powershell
# One-time: allow PowerShell to run local scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

.\run.ps1 setup
```

> 💡 To open PowerShell in your project folder: navigate to the folder in
> File Explorer → Shift + Right-click on empty space → **Open PowerShell window here**

**macOS / Linux:**
```bash
make setup
```

This creates a `.venv/` virtual environment and installs all dependencies into it.
You never need to activate the venv manually — all commands handle it automatically.

### 3. Get the dataset

Download the **Pima Indians Diabetes Dataset** from Kaggle:
[https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

Place the downloaded file at:
```
data/raw/diabetes.csv
```

### 4. Initialise DVC

**Windows:**
```powershell
.\run.ps1 dvc-init
```
**macOS / Linux:**
```bash
make dvc-init
```

### 5. Validate and train

**Windows:**
```powershell
.\run.ps1 validate   # cleans and validates the raw data
.\run.ps1 train      # trains all 3 models and logs everything to MLflow
                     # automatically sets the Production alias on the best model
```
**macOS / Linux:**
```bash
make validate
make train
```

At the end of training you will see something like:
```
✅ Alias 'Production' set on version 1 (logistic_regression)
```
No manual promotion step needed.

### 6. Evaluate the Production model

**Windows:**
```powershell
.\run.ps1 evaluate
```
**macOS / Linux:**
```bash
make evaluate
```

Results are printed to the terminal and saved to `metrics/metrics.json`.

### 7. Start the API and dashboard

**Windows:**
```powershell
.\run.ps1 serve
```
**macOS / Linux:**
```bash
make serve
```

| URL | What you get |
|---|---|
| [http://localhost:8000](http://localhost:8000) | Prediction dashboard |
| [http://localhost:8000/docs](http://localhost:8000/docs) | Swagger UI |
| [http://localhost:8000/health](http://localhost:8000/health) | Model load status |

---

## 🧪 Running Tests

**Windows:**
```powershell
.\run.ps1 test
```
**macOS / Linux:**
```bash
make test
```

Tests run without a live MLflow server — the model is fully mocked via `unittest.mock`.
11 tests covering API endpoints, data validation, and model training logic.

To run the linter:

**Windows:**
```powershell
.\run.ps1 lint
.\run.ps1 lint-fix   # auto-fix where possible
```
**macOS / Linux:**
```bash
make lint
make lint-fix
```

---

## 🔬 Viewing MLflow Experiment Runs

**Windows:**
```powershell
.\run.ps1 mlflow-ui
```
**macOS / Linux:**
```bash
make mlflow-ui
```

Open **http://127.0.0.1:5000** then click **diabetes-risk** in the left sidebar.

You will see all three model runs side by side with their metrics (AUC, F1, Accuracy).
The run with the `Production` alias tag is the one currently being served.

---

## 🌿 Optional: Connect DagsHub

DagsHub gives you a **hosted MLflow UI** with a public URL — useful for showing
recruiters your experiment runs without them having to run anything locally.

### Setup

1. Create a free account at [https://dagshub.com](https://dagshub.com)
2. Create a new repo and push your project to it
3. On DagsHub, go to your repo → **Remote → Experiments** to get your tracking URI
4. Copy `.env.example` to `.env` and fill in your credentials:

```
MLFLOW_TRACKING_URI=https://dagshub.com/<your_username>/<your_repo>.mlflow
MLFLOW_TRACKING_USERNAME=<your_dagshub_username>
MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>
```

5. Install the optional DagsHub dependency:

**Windows:**
```powershell
.venv\Scripts\pip install -e ".[dagshub]"
```
**macOS / Linux:**
```bash
.venv/bin/pip install -e ".[dagshub]"
```

6. Re-run training — all runs will now appear in your DagsHub experiment page:

**Windows:**
```powershell
.\run.ps1 train
```
**macOS / Linux:**
```bash
make train
```

> **Note:** The project works 100% locally without DagsHub.
> DagsHub is a bonus layer for sharing your experiment results publicly.

---

## 🐳 Docker

**Build the image:**

```bash
docker build -t diabetes-risk-mlops -f api/Dockerfile .
```

**Run the container:**

Windows (PowerShell):
```powershell
docker run -p 8000:8000 `
  -e MODEL_NAME=diabetes-risk-model `
  -e MLFLOW_TRACKING_URI=mlruns `
  -v ${PWD}/mlruns:/app/mlruns `
  diabetes-risk-mlops
```

macOS / Linux:
```bash
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
      # replace with your Docker Hub username
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

**Windows (PowerShell):**
```powershell
Invoke-RestMethod -Uri http://localhost:8000/predict `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"Pregnancies":2,"Glucose":138,"BloodPressure":72,"SkinThickness":35,"Insulin":80,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":47}'
```

**macOS / Linux:**
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

## 📋 Command Reference

| Action | Windows | macOS / Linux |
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