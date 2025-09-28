# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system deps (important for XGBoost)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
COPY combined_dataset1.csv .
COPY app.py .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose MLflow UI port
EXPOSE 5000

# Run MLflow tracking server + training
CMD mlflow ui --host 0.0.0.0 --port 5000 & python3 app.py



