.PHONY: setup dvc-init validate train evaluate serve test lint lint-fix clean

# Virtual environment location
VENV        := .venv
PYTHON      := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
MLRUNS_URI  := $(shell pwd)/mlruns

# ── Setup ─────────────────────────────────────────────────────────────────────
# Creates a local .venv, installs all dependencies into it, and copies .env
setup:
	python3 -m venv $(VENV)
	$(PIP) install -e ".[dev]"
	cp -n .env.example .env || true
	@echo ""
	@echo "✅ Setup complete. Virtual environment created at .venv/"
	@echo ""
	@echo "All make commands use .venv automatically — no need to activate it."
	@echo "If you want a shell inside the venv, run: source .venv/bin/activate"
	@echo ""
	@echo "Edit .env if you want to use DagsHub for MLflow tracking."

# ── DVC ───────────────────────────────────────────────────────────────────────
dvc-init:
	$(PYTHON) -m dvc init
	mkdir -p .dvc/remote/local_storage
	$(PYTHON) -m dvc remote add -d local_remote .dvc/remote/local_storage
	$(PYTHON) -m dvc add data/raw/diabetes.csv
	@echo "DVC initialised. Run 'make dvc-push' to push data to remote."

dvc-push:
	$(PYTHON) -m dvc push

dvc-pull:
	$(PYTHON) -m dvc pull

# ── Pipeline ──────────────────────────────────────────────────────────────────
validate:
	$(PYTHON) src/validate.py

train:
	MLFLOW_TRACKING_URI=$(MLRUNS_URI) $(PYTHON) src/train.py

evaluate:
	MLFLOW_TRACKING_URI=$(MLRUNS_URI) $(PYTHON) src/evaluate.py

# Run full DVC pipeline (validate → train → evaluate)
pipeline:
	$(PYTHON) -m dvc repro

# ── Serving ───────────────────────────────────────────────────────────────────
serve:
	MLFLOW_TRACKING_URI=$(MLRUNS_URI) $(VENV)/bin/uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

mlflow-ui:
	$(VENV)/bin/mlflow ui --backend-store-uri $(MLRUNS_URI) --host 127.0.0.1 --port 5000

# ── Testing & Linting ─────────────────────────────────────────────────────────
test:
	$(VENV)/bin/pytest tests/ -v --tb=short

lint:
	$(VENV)/bin/ruff check src/ api/ tests/

lint-fix:
	$(VENV)/bin/ruff check --fix src/ api/ tests/

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f mlruns_ci.db
	@echo "Cleaned."

# Removes the virtual environment entirely (re-run make setup to rebuild)
clean-venv: clean
	rm -rf $(VENV)
	@echo "Virtual environment removed. Run 'make setup' to recreate it."
