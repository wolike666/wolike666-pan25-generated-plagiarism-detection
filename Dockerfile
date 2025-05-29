FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Copy only pip requirements first (for better cache)
COPY requirements.txt .

# Install Python dependencies
RUN apt-get update && apt-get install -y python3-pip && pip3 install --no-cache-dir --break-system-packages -r requirements.txt

RUN python3 -c 'import nltk; nltk.download("punkt_tab")'

# Copy application code
COPY run.py .
COPY scripts/ scripts/

# Ensure entrypoint script is executable (if needed)
RUN chmod +x run.py

# Default command: TIRA will mount $inputDataset→/input and $outputDir→/output
ENTRYPOINT ["python", "run.py", "--input", "/input", "--output", "/output"]
