# Use a slim Python base image
FROM python:3.9-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Copy only pip requirements first (for better cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY run.py .
COPY scripts/ scripts/

# Ensure entrypoint script is executable (if needed)
RUN chmod +x run.py

# Default command: TIRA will mount $inputDataset→/input and $outputDir→/output
ENTRYPOINT ["python", "run.py", "--input", "/input", "--output", "/output"]
