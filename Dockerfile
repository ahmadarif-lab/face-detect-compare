FROM python:3.9-slim

WORKDIR /app

# System deps for opencv
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads

ENV PORT=7860
EXPOSE 7860

# 1 worker supaya model tidak di-load berkali-kali, timeout panjang untuk inference ML
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "300", "app:app"]
