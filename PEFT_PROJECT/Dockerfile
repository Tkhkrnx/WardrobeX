FROM python:3.12-slim

WORKDIR /app

# 使用清华源 + 安装 OpenCV 常用依赖
RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/debian bookworm main contrib non-free\n\
deb http://mirrors.tuna.tsinghua.edu.cn/debian bookworm-updates main contrib non-free\n\
deb http://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main contrib non-free\n" \
> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgtk2.0-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# 用清华源安装 Python 依赖
RUN pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
