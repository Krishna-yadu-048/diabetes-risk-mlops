# run.ps1 — Windows equivalent of the Makefile
# Usage: .\run.ps1 <command>
# Example: .\run.ps1 setup

param(
    [Parameter(Mandatory=$true)]
    [string]$Command
)

# ── Paths ─────────────────────────────────────────────────────────────────────
$VENV       = ".venv"
$PYTHON     = "$VENV\Scripts\python.exe"
$PIP        = "$VENV\Scripts\pip.exe"
$UVICORN    = "$VENV\Scripts\uvicorn.exe"
$MLFLOW     = "$VENV\Scripts\mlflow.exe"
$PYTEST     = "$VENV\Scripts\pytest.exe"
$RUFF       = "$VENV\Scripts\ruff.exe"
$MLRUNS_PATH = "$PSScriptRoot\mlruns"
# MLflow on Windows requires the file:/// URI scheme for the model registry
$MLRUNS_URI  = "file:///" + $MLRUNS_PATH.Replace("\", "/")

# ── Helper ────────────────────────────────────────────────────────────────────
function Run($cmd) {
    Write-Host "> $cmd" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Command failed with exit code $LASTEXITCODE" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# ── Commands ──────────────────────────────────────────────────────────────────
switch ($Command) {

    "setup" {
        Write-Host "Creating virtual environment..." -ForegroundColor Green
        Run "python -m venv $VENV"
        Run "& '$PIP' install -e '.[dev]'"
        if (-not (Test-Path ".env")) {
            Copy-Item ".env.example" ".env"
            Write-Host "Copied .env.example to .env" -ForegroundColor Yellow
        }
        Write-Host ""
        Write-Host "Setup complete. Virtual environment created at .venv\" -ForegroundColor Green
        Write-Host "Run commands with: .\run.ps1 <command>"
        Write-Host "Edit .env if you want to use DagsHub for MLflow tracking."
    }

    "dvc-init" {
        Run "& '$PYTHON' -m dvc init"
        New-Item -ItemType Directory -Force -Path ".dvc\remote\local_storage" | Out-Null
        Run "& '$PYTHON' -m dvc remote add -d local_remote .dvc\remote\local_storage"
        Run "& '$PYTHON' -m dvc add data\raw\diabetes.csv"
        Write-Host "DVC initialised. Run '.\run.ps1 dvc-push' to push data to remote."
    }

    "dvc-push" {
        Run "& '$PYTHON' -m dvc push"
    }

    "dvc-pull" {
        Run "& '$PYTHON' -m dvc pull"
    }

    "validate" {
        Run "& '$PYTHON' src\validate.py"
    }

    "train" {
        $env:MLFLOW_TRACKING_URI = $MLRUNS_URI
        Run "& '$PYTHON' src\train.py"
    }

    "evaluate" {
        $env:MLFLOW_TRACKING_URI = $MLRUNS_URI
        Run "& '$PYTHON' src\evaluate.py"
    }

    "pipeline" {
        Run "& '$PYTHON' -m dvc repro"
    }

    "serve" {
        $env:MLFLOW_TRACKING_URI = $MLRUNS_URI
        Run "& '$UVICORN' api.main:app --reload --host 0.0.0.0 --port 8000"
    }

    "mlflow-ui" {
        Run "& '$MLFLOW' ui --backend-store-uri '$MLRUNS_URI' --host 127.0.0.1 --port 5000"
    }

    "test" {
        Run "& '$PYTEST' tests\ -v --tb=short"
    }

    "lint" {
        Run "& '$RUFF' check src\ api\ tests\"
    }

    "lint-fix" {
        Run "& '$RUFF' check --fix src\ api\ tests\"
    }

    "clean" {
        Write-Host "Cleaning up..." -ForegroundColor Yellow
        Get-ChildItem -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Recurse -Filter ".pytest_cache" -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
        if (Test-Path "mlruns_ci.db") { Remove-Item "mlruns_ci.db" -Force }
        Write-Host "Cleaned." -ForegroundColor Green
    }

    "clean-venv" {
        .\run.ps1 clean
        if (Test-Path $VENV) {
            Remove-Item $VENV -Recurse -Force
            Write-Host "Virtual environment removed. Run '.\run.ps1 setup' to recreate it." -ForegroundColor Yellow
        }
    }

    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host ""
        Write-Host "Available commands:" -ForegroundColor Cyan
        Write-Host "  setup        Create .venv and install dependencies"
        Write-Host "  dvc-init     Initialise DVC with a local remote"
        Write-Host "  dvc-push     Push data to DVC remote"
        Write-Host "  dvc-pull     Pull data from DVC remote"
        Write-Host "  validate     Run data validation and cleaning"
        Write-Host "  train        Train all 3 models and log to MLflow"
        Write-Host "  evaluate     Evaluate Production model on test set"
        Write-Host "  pipeline     Run full DVC pipeline"
        Write-Host "  serve        Start FastAPI at http://localhost:8000"
        Write-Host "  mlflow-ui    Start MLflow UI at http://127.0.0.1:5000"
        Write-Host "  test         Run all pytest tests"
        Write-Host "  lint         Run ruff linter"
        Write-Host "  lint-fix     Auto-fix lint issues"
        Write-Host "  clean        Remove __pycache__, .pytest_cache, etc."
        Write-Host "  clean-venv   Remove .venv entirely"
        exit 1
    }
}