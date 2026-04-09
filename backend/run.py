"""阿斯拉量化系統 — 後端啟動腳本（自動 Port 管理版）"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests
import uvicorn

from app.core.config.settings import settings

# Port 掃描範圍
_PORT_RANGE_START = 8000
_PORT_RANGE_END = 8010  # 最多嘗試 11 個 port
_PORT_FILE = Path(__file__).parent / ".port"
_APP_IDENTIFIER = "阿斯拉量化系統"


def _find_pid_on_port(port: int) -> int | None:
    """找出佔用指定 port 的進程 PID（macOS / Linux）"""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _is_our_app(port: int) -> bool:
    """檢查指定 port 上跑的是否為阿斯拉量化系統"""
    try:
        resp = requests.get(f"http://127.0.0.1:{port}/api/health", timeout=2)
        data = resp.json()
        # 檢查回應中是否包含我們的系統名稱
        return _APP_IDENTIFIER in data.get("system", "")
    except Exception:
        return False


def _kill_process_on_port(port: int) -> bool:
    """終止佔用 port 的進程，回傳是否成功"""
    pid = _find_pid_on_port(port)
    if pid is None or pid == os.getpid():
        return False

    print(f"  ⚠️  終止舊的阿斯拉進程 (PID {pid}) ...")
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(30):
            if not _is_port_in_use(port):
                print(f"  ✅ 舊進程已終止，Port {port} 已釋放")
                return True
            time.sleep(0.1)
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        print(f"  ✅ 舊進程已強制終止")
        return True
    except ProcessLookupError:
        return True


def _resolve_port() -> int:
    """智慧 Port 解析：自動找到可用的 port。

    優先順序：
    1. 上次成功使用的 port（從 .port 檔讀取）→ 若仍可用直接使用
    2. 設定檔中的預設 port（settings.port）
    3. 自動掃描 8000-8010 找到可用 port

    若 port 被阿斯拉自己佔用 → 接管（kill 舊進程）
    若 port 被其他系統佔用 → 跳過，嘗試下一個
    """
    # 收集候選 port（去重，保持順序）
    candidates: list[int] = []

    # 優先：上次成功的 port
    if _PORT_FILE.exists():
        try:
            last_port = int(_PORT_FILE.read_text().strip())
            if _PORT_RANGE_START <= last_port <= _PORT_RANGE_END:
                candidates.append(last_port)
        except (ValueError, OSError):
            pass

    # 其次：設定檔預設
    if settings.port not in candidates:
        candidates.append(settings.port)

    # 最後：掃描範圍內所有 port
    for p in range(_PORT_RANGE_START, _PORT_RANGE_END + 1):
        if p not in candidates:
            candidates.append(p)

    occupied_by_others: list[tuple[int, int | None]] = []

    for port in candidates:
        if not _is_port_in_use(port):
            print(f"  ✅ Port {port} 可用")
            return port

        # Port 被佔用，檢查是不是我們的 app
        if _is_our_app(port):
            print(f"  🔄 Port {port} 被舊的阿斯拉進程佔用，正在接管...")
            if _kill_process_on_port(port):
                return port

        # 被其他系統佔用，記錄並跳過
        pid = _find_pid_on_port(port)
        occupied_by_others.append((port, pid))
        print(f"  ⏭️  Port {port} 被其他程式佔用 (PID {pid})，跳過")

    # 所有 port 都被佔用
    print(f"\n❌ Port {_PORT_RANGE_START}-{_PORT_RANGE_END} 全部被佔用：")
    for port, pid in occupied_by_others:
        pid_info = f"PID {pid}" if pid else "未知進程"
        print(f"   Port {port}: {pid_info}")
    print(f"\n💡 解決方式：")
    print(f"   1. 手動釋放一個 port: kill <PID>")
    print(f"   2. 或設定環境變數指定其他 port: PORT=9000 python run.py")
    sys.exit(1)


def _write_port_file(port: int):
    """將實際使用的 port 寫入 .port 檔，供前端 vite 讀取"""
    try:
        _PORT_FILE.write_text(str(port))
        print(f"  📝 Port 已寫入 {_PORT_FILE}")
    except OSError as e:
        print(f"  ⚠️  無法寫入 .port 檔: {e}")


if __name__ == "__main__":
    print("🚀 阿斯拉量化系統 — 啟動中...")
    print(f"  預設 Port: {settings.port} | 掃描範圍: {_PORT_RANGE_START}-{_PORT_RANGE_END}")

    port = _resolve_port()
    _write_port_file(port)

    print(f"\n🌐 後端啟動於 http://{settings.host}:{port}")
    print(f"   前端請確保 proxy 指向此 port（vite.config.ts 已設定自動讀取）\n")

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
