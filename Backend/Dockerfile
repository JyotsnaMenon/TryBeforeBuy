# 1. Use an official, lightweight Python 3.11 image as the base
FROM python:3.11-slim

# 2. Set environment variables for better Python performance in Docker
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install essential Linux system drivers for OpenCV and DeepFace
# This prevents the "libgl1" and "libxcb" errors on cloud servers
# 4. Install essential Linux system drivers for OpenCV and DeepFace
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy ONLY the requirements file first (to leverage Docker caching)
COPY requirements.txt .

# 6. Install the Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of your Backend code (app.py, models folder, dataset.json)
COPY . .

# 8. Expose the port so the cloud provider can route traffic to it
EXPOSE $PORT

# 9. Start the server using Gunicorn for production-grade concurrency
CMD gunicorn app:app --workers 1 --threads 3 --bind 0.0.0.0:$PORT --timeout 120

