# stages backend
FROM python:3.9-slim AS backend

# Install git untuk meng-clone repository
RUN apt-get update && apt-get install -y git

# Buat working directory untuk backend
WORKDIR /backend

# Clone branch backend dari repository
RUN git clone -b backend https://github.com/mhdusop/GAN-Skin-Disease-Identifier.git .

# Tampilkan daftar file untuk verifikasi
RUN ls -l /backend

# Salin requirements.txt dari konteks build ke dalam container
COPY requirements.txt requirements.txt

# Install dependencies dari requirements.txt
RUN pip install -r requirements.txt

# Salin semua file ke dalam container
COPY . .

# Expose port 5000 untuk backend
EXPOSE 5000

# Jalankan aplikasi backend dengan Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
