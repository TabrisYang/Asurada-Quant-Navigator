"""阿斯拉量化系統 — SQLite 資料庫自動備份。

策略：
- Hot backup（用 SQLite .backup API，不會複製到寫一半的損壞狀態）
- 跳過純快取 DB（analysis_cache / semantic_cache，可重生）
- GFS 分層保留（無年度永久版）：每天 7 + 每週日 4 + 每月 1 號 12（容量上限 ~230 MB）

執行方式：
- 手動：python3 backup_databases.py
- 自動：launchd 每天 0:00（透過 install_backup_launchd.sh 安裝）
"""

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_DB_DIR = _SCRIPT_DIR.parent / "data" / "db"
_BACKUP_ROOT = _DB_DIR / "backups"
_SKIP_PATTERNS = ("analysis_cache", "semantic_cache")  # 純快取，可重生

_KEEP_DAILY = 7      # 最近 7 天每天保留
_KEEP_WEEKLY = 4     # 最近 4 個週日多保留
_KEEP_MONTHLY = 12   # 最近 12 個月初多保留
# 「今天跑過就跳過」旗標檔（避免 RunAtLoad 重複觸發）
_LAST_RUN_FILE = _DB_DIR / ".backup_last_run"


def hot_backup(src: Path, dst: Path) -> bool:
    """SQLite hot backup：transaction 一致，即使 src 正在被寫入也安全。"""
    try:
        src_conn = sqlite3.connect(str(src))
        dst_conn = sqlite3.connect(str(dst))
        src_conn.backup(dst_conn)
        dst_conn.close()
        src_conn.close()
        return True
    except Exception as e:
        print(f"  ✗ {src.name} 備份失敗：{e}")
        return False


def backup_today() -> Path | None:
    today = datetime.now().strftime("%Y-%m-%d")
    target_dir = _BACKUP_ROOT / today
    target_dir.mkdir(parents=True, exist_ok=True)

    db_files = sorted(_DB_DIR.glob("*.db"))
    if not db_files:
        print(f"⚠ 找不到任何 .db 檔案於 {_DB_DIR}")
        return None

    print(f"📦 備份至 {target_dir}")
    backed = 0
    skipped = 0
    failed = 0
    for db in db_files:
        if any(p in db.name for p in _SKIP_PATTERNS):
            print(f"  ⊘ 跳過 {db.name}（純快取，可重生）")
            skipped += 1
            continue
        if hot_backup(db, target_dir / db.name):
            size_kb = (target_dir / db.name).stat().st_size // 1024
            print(f"  ✓ {db.name} ({size_kb} KB)")
            backed += 1
        else:
            failed += 1

    print(f"\n結果：成功 {backed} ｜ 跳過 {skipped} ｜ 失敗 {failed}")
    return target_dir if backed > 0 else None


def cleanup_old_backups():
    """套用 GFS 分層保留策略。"""
    if not _BACKUP_ROOT.exists():
        return

    backup_dirs = sorted(
        [d for d in _BACKUP_ROOT.iterdir() if d.is_dir()],
        reverse=True,
    )
    if not backup_dirs:
        return

    today = datetime.now().date()
    keep: set[str] = set()
    sundays_kept = 0
    months_kept = 0

    for d in backup_dirs:
        try:
            dt = datetime.strptime(d.name, "%Y-%m-%d").date()
        except ValueError:
            keep.add(d.name)  # 不認識的目錄保留以防誤刪
            continue

        # 規則 1：最近 _KEEP_DAILY 天每天保留
        if (today - dt).days < _KEEP_DAILY:
            keep.add(d.name)

        # 規則 2：最近 _KEEP_WEEKLY 個週日
        if dt.weekday() == 6 and sundays_kept < _KEEP_WEEKLY:
            keep.add(d.name)
            sundays_kept += 1

        # 規則 3：最近 _KEEP_MONTHLY 個月初
        if dt.day == 1 and months_kept < _KEEP_MONTHLY:
            keep.add(d.name)
            months_kept += 1

        # 規則 4（已移除）：原本每年 1/1 永久保留 — 改為「無年度永久版」
        # 容量上限固定 7+4+12 = 23 份 ~230 MB，永久不長

    deleted = 0
    for d in backup_dirs:
        if d.name not in keep:
            shutil.rmtree(d)
            deleted += 1
    if deleted > 0:
        print(f"\n🧹 清理 {deleted} 個過期備份（保留 {len(keep)} 個）")


def main():
    print("🗄  阿斯拉量化系統 — SQLite 自動備份")
    print(f"   時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 「今天已跑過就跳過」邏輯（避免 RunAtLoad 在已跑過的同日重複觸發）
    today_str = datetime.now().date().isoformat()
    if _LAST_RUN_FILE.exists():
        try:
            last_str = _LAST_RUN_FILE.read_text().strip()
            if last_str == today_str:
                print(f"⊘ Backup 今天 ({today_str}) 已跑過，跳過")
                return
        except Exception:
            pass

    target = backup_today()
    if target is None:
        sys.exit(1)

    cleanup_old_backups()
    # 寫入「今天跑過」旗標
    try:
        _LAST_RUN_FILE.write_text(today_str)
    except Exception:
        pass
    print(f"\n✅ 完成：{target}")


if __name__ == "__main__":
    main()
