# 1. Stay with 3.11-slim, it's perfect for 1GB limits
FROM python:3.11-slim

# 2. Performance and Port settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Railway will override this $PORT, but 5000 is a good local default
ENV PORT=5000 

# 3. Working directory
WORKDIR /app

# 4. Leaner System Drivers
# We only need the core GL and GLib libraries for OpenCV. 
# Removing the DeepFace-specific X11/Render libraries to save space.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Requirement Caching
COPY requirements.txt .

# 6. Install Python dependencies
# Adding --no-cache-dir is CRITICAL for Railway to prevent "Disk Limit Reached"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copy Backend code (including models/ and dataset.json)
COPY . .

# 8. Use the $PORT variable Railway provides
EXPOSE $PORT

# 9. Gunicorn Start Command
# Kept at 1 worker and 2-3 threads. This is the "Sweet Spot" for 1GB RAM.
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 3 --timeout 120