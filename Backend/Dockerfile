# 1. Use Python 3.11-slim
FROM python:3.11-slim

# 2. Performance and Port settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000 

# 3. Working directory inside the container
WORKDIR /app

# 4. FIXED: Modern Leaner System Drivers
# Using libgl1 instead of the old mesa-glx naming
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Requirement Caching
COPY requirements.txt .

# 6. Install Python dependencies
# This will now skip the heavy DeepFace/TensorFlow stuff
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copy Backend code (including models/ and dataset.json)
COPY . .

# 8. Use the $PORT variable Railway provides
EXPOSE $PORT

# 9. Gunicorn Start Command
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 3 --timeout 120