FROM python:3.11-slim

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV OPENSSL_CONF=/etc/ssl/openssl.cnf

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    openssl \
    wget \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# OpenSSL config
RUN echo "openssl_conf = openssl_init" > /etc/ssl/openssl.cnf && \
    echo "" >> /etc/ssl/openssl.cnf && \
    echo "[openssl_init]" >> /etc/ssl/openssl.cnf && \
    echo "ssl_conf = ssl_sect" >> /etc/ssl/openssl.cnf && \
    echo "" >> /etc/ssl/openssl.cnf && \
    echo "[ssl_sect]" >> /etc/ssl/openssl.cnf && \
    echo "system_default = system_default_sect" >> /etc/ssl/openssl.cnf && \
    echo "" >> /etc/ssl/openssl.cnf && \
    echo "[system_default_sect]" >> /etc/ssl/openssl.cnf && \
    echo "CipherString = DEFAULT:@SECLEVEL=1" >> /etc/ssl/openssl.cnf

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p /tmp /app/sessions

# Expose port
EXPOSE 8000

# Create startup script
RUN echo '#!/bin/bash\n\
echo "🚀 Starting SK4FiLM System..."\n\
echo "🤖 Starting Bot..."\n\
python bot.py &\n\
BOT_PID=$!\n\
echo "Bot PID: $BOT_PID"\n\
sleep 5\n\
echo "🌐 Starting Web Server..."\n\
python main.py &\n\
WEB_PID=$!\n\
echo "Web PID: $WEB_PID"\n\
echo "✅ All services started!"\n\
wait -n\n\
exit $?' > /app/start.sh && chmod +x /app/start.sh

# Start with script
CMD ["/bin/bash", "/app/start.sh"]
