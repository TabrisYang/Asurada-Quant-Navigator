# === 阿斯拉量化系統 — Multi-stage Dockerfile ===

# ── Stage 1: 前端建置 ──
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: 後端 + 靜態檔 ──
FROM python:3.12-slim AS runtime
WORKDIR /app

# 系統依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python 依賴
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 後端程式碼
COPY backend/ ./backend/

# 前端靜態檔（由 Stage 1 產出）
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# 資料目錄
RUN mkdir -p /app/data

ENV PYTHONPATH=/app/backend
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
