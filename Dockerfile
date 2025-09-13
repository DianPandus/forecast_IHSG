# ---- Base ----
FROM python:3.10-slim

# Env supaya build cepat/bersih
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Optional: dependensi sistem (uncomment kalau perlu)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential gcc g++ \
#  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies lebih cepat dengan layer terpisah
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt && pip install jupyter nbconvert

# Default command: shell (supaya bisa override di `docker run`)
CMD ["bash"]
