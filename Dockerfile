# --- STAGE 1: Builder (The "Heavy" stage) ---
FROM python:3.10-slim as builder

WORKDIR /app

# Install compilers needed for psutil/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install packages to a local user directory
RUN pip install --user --no-cache-dir -r requirements.txt


# --- STAGE 2: Final (The "Lightweight" stage) ---
FROM python:3.10-slim

# Standard environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/root/.local/bin:$PATH

WORKDIR /app

# Copy only the final installed libraries from the builder
COPY --from=builder /root/.local /root/.local

# Copy your code
COPY *.py .

# Ensure the shared volume directory exists
RUN mkdir -p /app/runs

EXPOSE 8080
EXPOSE 8501

CMD ["python", "server.py"]
