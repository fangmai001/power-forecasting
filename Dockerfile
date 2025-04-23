# 構建階段
FROM python:3.9-slim as builder

# 設置工作目錄
WORKDIR /app

# 安裝構建依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 複製依賴文件
COPY requirements.txt .

# 創建虛擬環境並安裝依賴
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# 運行階段
FROM python:3.9-slim

# 創建非 root 用戶
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown appuser:appuser /app

# 從構建階段複製虛擬環境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 設置工作目錄
WORKDIR /app

# 複製應用代碼
COPY --chown=appuser:appuser . /app/

# 切換到非 root 用戶
USER appuser

# 設置環境變量
ENV PYTHONPATH=/app

# 健康檢查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# # 設置默認命令
# CMD ["python", "main.py"] 