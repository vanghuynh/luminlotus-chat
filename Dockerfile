# Sử dụng Python image nhẹ
FROM python:3.10-slim

# Cài đặt biến môi trường để tránh gợi ý input
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Cập nhật và cài đặt gói hệ thống cơ bản
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Tạo thư mục app và đặt làm thư mục làm việc
WORKDIR /app

# Sao chép toàn bộ code vào container
COPY . .

# Cài đặt thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Mở cổng Flask
EXPOSE 5000

# Chạy ứng dụng Flask
CMD ["python", "app.py"]
