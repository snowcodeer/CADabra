FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV DISPLAY=:99
CMD Xvfb :99 -screen 0 1024x768x24 & uvicorn main:app --host 0.0.0.0 --port 8000
