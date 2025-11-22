# Use a slim Python 3.11 image to keep the container small
FROM python:3.11-slim

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirement list and install dependencies first for better caching
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# Default command runs the CLI with angle encoding feature map
CMD ["python", "app/main.py", "--feature-map", "angle_encoding"]
