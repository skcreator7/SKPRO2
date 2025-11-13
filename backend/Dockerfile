FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ ca-certificates libssl-dev openssl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy application
COPY app.py Procfile ./
RUN mkdir -p /tmp

EXPOSE 8000

CMD ["python", "app.py"]
