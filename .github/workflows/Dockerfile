# Dockerfile

# Gunakan base image Python yang ringan
FROM python:3.10-slim

# Tetapkan folder kerja di dalam container
WORKDIR /app

# Salin file requirements terlebih dahulu untuk caching
COPY requirements.txt .

# Install semua library yang dibutuhkan
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir jupyter nbconvert

# Salin semua file proyek ke dalam folder kerja
COPY . .

# Perintah default yang akan dijalankan saat container启动
CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "Model.ipynb"]