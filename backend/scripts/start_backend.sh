#!/bin/bash
# 啟動 Asurada backend（背景模式 + log + PID 管理）
#
# 用法：
#   bash backend/scripts/start_backend.sh         # 啟動
#   bash backend/scripts/start_backend.sh stop    # 停止
#   bash backend/scripts/start_backend.sh status  # 查看狀態
#   bash backend/scripts/start_backend.sh restart # 重啟

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PIDFILE=/tmp/asurada_backend.pid
LOGFILE=/tmp/asurada_backend.log

ACTION="${1:-start}"

_is_running() {
    [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null
}

_start() {
    if _is_running; then
        echo "✅ Backend 已在跑（PID $(cat $PIDFILE)，port 8000）"
        return 0
    fi

    if [ ! -x "$BACKEND_DIR/.venv/bin/python3" ]; then
        echo "❌ 找不到 .venv/bin/python3，請先 cd backend && python3 -m venv .venv"
        exit 1
    fi

    cd "$BACKEND_DIR"
    # ★ v101 修復：限制 OpenMP / MKL 執行緒，避免 ML 庫同時搶 native threads → segfault
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export KMP_DUPLICATE_LIB_OK=TRUE
    nohup .venv/bin/python3 -m uvicorn app.main:app \
        --host 0.0.0.0 --port 8000 \
        > "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 2

    if _is_running; then
        echo "✅ Backend 已啟動（PID $(cat $PIDFILE)，port 8000）"
        echo "   log: $LOGFILE"
        echo "   驗證：curl http://localhost:8000/api/predictions/imitation/status"
    else
        echo "❌ Backend 啟動失敗，看 $LOGFILE"
        tail -20 "$LOGFILE"
        exit 1
    fi
}

_stop() {
    if ! _is_running; then
        echo "⚠ Backend 未在跑"
        return 0
    fi

    PID=$(cat "$PIDFILE")
    kill -TERM "$PID" 2>/dev/null || true
    sleep 1
    if _is_running; then
        kill -KILL "$PID" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
    echo "✅ Backend 已停止"
}

_status() {
    if _is_running; then
        echo "✅ Backend 在跑（PID $(cat $PIDFILE)，port 8000）"
        echo "   log: tail -f $LOGFILE"
    else
        echo "○ Backend 未在跑"
    fi
}

case "$ACTION" in
    start) _start ;;
    stop) _stop ;;
    restart) _stop; sleep 1; _start ;;
    status) _status ;;
    *) echo "用法：$0 {start|stop|restart|status}"; exit 1 ;;
esac
