# Multi-stage production Dockerfile for ML Model Serving & Monitoring API
# Stage 1: Build & Dependency Installation
FROM python:3.11-slim AS builder

WORKDIR /build

# Install security updates and build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Hardened Runtime Container
FROM python:3.11-slim AS runner

WORKDIR /app

# Create non-root system user for security compliance (Principle of Least Privilege)
RUN groupadd -g 10001 mlgroup && \
    useradd -u 10001 -g mlgroup -s /bin/bash -m mluser

# Install runtime curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder stage
COPY --from=builder /install /usr/local

# Copy application source code and baseline artifacts
COPY src/ /app/src/
COPY data/ /app/data/
COPY artifacts/ /app/artifacts/
COPY scripts/ /app/scripts/

# Set correct ownership for non-root user
RUN chown -R mluser:mlgroup /app

# Switch to non-root user
USER mluser

# Environment configurations
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    ARTIFACTS_DIR=/app/artifacts

EXPOSE 8000

# Docker Healthcheck
HEALTHCHECK --interval=10s --timeout=3s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production ASGI server
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]
