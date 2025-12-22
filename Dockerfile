FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install uv for package management
RUN pip install --no-cache-dir uv

# Copy application code first so local package is available
COPY . .

# Install dependencies
RUN uv sync --frozen

# Set Python path
ENV PYTHONPATH=/app/src

# Default command (can override for CLI)
CMD ["uv", "run", "board", "--help"]
