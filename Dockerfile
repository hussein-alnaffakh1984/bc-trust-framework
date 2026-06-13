# Reproducible environment for the full experiments.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Default: run the fast smoke tests. Override to run an analysis script, e.g.:
#   docker run --rm -v $PWD/data:/app/data IMAGE python src/multicohort_evaluation.py
CMD ["pytest", "-q"]
