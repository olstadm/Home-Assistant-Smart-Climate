ARG BUILD_FROM=ghcr.io/hassio-addons/debian-base:7.2.0
FROM $BUILD_FROM

# Set shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install Python and required system packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3=3.11.* \
        python3-pip=23.0.* \
        python3-dev \
        gcc \
        g++ \
        curl \
        jq \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages --upgrade pip \
    && pip3 install --no-cache-dir --break-system-packages -r requirements.txt \
    && pip3 install --no-cache-dir --break-system-packages \
        aiohttp==3.8.* \
        aiofiles==23.* \
        pyyaml==6.*

# Copy application files
COPY smart_climate/ ./smart_climate/
COPY run.sh .
COPY config.json .

# Make run script executable
RUN chmod +x run.sh

# Create data directory for persistent storage
RUN mkdir -p /data/models

# Set environment variables
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Expose port (optional API endpoint)
EXPOSE 8099

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8099/health || exit 1

# Run the service
CMD ["/app/run.sh"]