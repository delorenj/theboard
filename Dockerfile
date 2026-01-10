FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for package management
RUN pip install --no-cache-dir uv

# Copy Bloodbank dependency first (from sibling directory)
COPY bloodbank/trunk-main /bloodbank

# Copy application code
COPY theboard/trunk-main .

# Install dependencies (Bloodbank will be installed from /bloodbank)
RUN uv sync --frozen

# Set Python path
ENV PYTHONPATH=/app/src

# Default command (can override for CLI)
CMD ["uv", "run", "board", "--help"]
