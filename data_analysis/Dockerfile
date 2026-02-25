FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for matplotlib & opencv
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy entire analysis directory
COPY . .

# Default command
CMD ["python", "src/main.py"]