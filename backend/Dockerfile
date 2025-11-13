FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OPENSSL_CONF=/etc/ssl/openssl.cnf

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ ca-certificates libssl-dev libffi-dev openssl wget procps \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# OpenSSL config
RUN echo "openssl_conf = openssl_init" > /etc/ssl/openssl.cnf && \
    echo "[openssl_init]" >> /etc/ssl/openssl.cnf && \
    echo "ssl_conf = ssl_sect" >> /etc/ssl/openssl.cnf && \
    echo "[ssl_sect]" >> /etc/ssl/openssl.cnf && \
    echo "system_default = system_default_sect" >> /etc/ssl/openssl.cnf && \
    echo "[system_default_sect]" >> /etc/ssl/openssl.cnf && \
    echo "CipherString = DEFAULT:@SECLEVEL=1" >> /etc/ssl/openssl.cnf

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY bot.py main.py start.sh Procfile ./
RUN chmod +x start.sh

# Create dirs
RUN mkdir -p /tmp /app/sessions

EXPOSE 8000

# CRITICAL: Explicit run command
CMD ["bash", "start.sh"]
