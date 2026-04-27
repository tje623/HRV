#!/usr/bin/env python3
"""
Run a command with durable logging and a JSON manifest.

Example:
    python Scripts/utils/run_audit.py \
      --name beat_features \
      --log-dir Docs/run_logs \
      --input Data/Processed/peaks.parquet \
      --output Data/Processed/beat_features.parquet \
      -- python Scripts/features/beat_features.py --processed-dir Data/Processed
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _run_text(cmd: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _git_info(cwd: Path) -> dict[str, object]:
    root = _run_text(["git", "rev-parse", "--show-toplevel"], cwd)
    if not root:
        return {"available": False}
    root_path = Path(root)
    status = _run_text(["git", "status", "--short"], root_path)
    return {
        "available": True,
        "root": str(root_path),
        "commit": _run_text(["git", "rev-parse", "HEAD"], root_path),
        "branch": _run_text(["git", "rev-parse", "--abbrev-ref", "HEAD"], root_path),
        "status_short": status or "",
        "dirty": bool(status),
    }


def _path_info(path: str) -> dict[str, object]:
    p = Path(path)
    exists = p.exists()
    info: dict[str, object] = {
        "path": str(p),
        "exists": exists,
    }
    if exists:
        st = p.stat()
        info.update({
            "size_bytes": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(
                timespec="seconds"
            ),
        })
    return info


def _reader_thread(src: TextIO, terminal: TextIO, log: TextIO) -> None:
    for line in src:
        terminal.write(line)
        terminal.flush()
        log.write(line)
        log.flush()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrap a script run with terminal logs and a JSON manifest.",
    )
    parser.add_argument("--name", required=True, help="Short run name used in filenames")
    parser.add_argument("--log-dir", default="Docs/run_logs", help="Directory for logs")
    parser.add_argument("--input", action="append", default=[], help="Input path to record")
    parser.add_argument("--output", action="append", default=[], help="Output path to record")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Environment variable name to record if present",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run, usually after --",
    )
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("No command supplied. Put the command after --.")
    return args


def main() -> int:
    args = _parse_args()
    cwd = Path.cwd()
    started_at = _utc_now()
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in args.name)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    prefix = log_dir / f"{run_stamp}_{safe_name}"
    stdout_path = prefix.with_suffix(".stdout.log")
    stderr_path = prefix.with_suffix(".stderr.log")
    manifest_path = prefix.with_suffix(".manifest.json")

    manifest: dict[str, object] = {
        "name": args.name,
        "started_at_utc": started_at,
        "cwd": str(cwd),
        "command": args.command,
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "git": _git_info(cwd),
        "env": {name: os.environ.get(name) for name in args.env},
        "inputs_before": [_path_info(p) for p in args.input],
        "outputs_before": [_path_info(p) for p in args.output],
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }

    print(f"[run_audit] manifest: {manifest_path}")
    print(f"[run_audit] stdout:   {stdout_path}")
    print(f"[run_audit] stderr:   {stderr_path}")
    print(f"[run_audit] command:  {' '.join(args.command)}")

    with stdout_path.open("w", encoding="utf-8") as stdout_log, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr_log:
        proc = subprocess.Popen(
            args.command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        assert proc.stdout is not None and proc.stderr is not None
        threads = [
            threading.Thread(
                target=_reader_thread,
                args=(proc.stdout, sys.stdout, stdout_log),
                daemon=True,
            ),
            threading.Thread(
                target=_reader_thread,
                args=(proc.stderr, sys.stderr, stderr_log),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        returncode = proc.wait()
        for thread in threads:
            thread.join()

    manifest.update({
        "finished_at_utc": _utc_now(),
        "exit_code": returncode,
        "inputs_after": [_path_info(p) for p in args.input],
        "outputs_after": [_path_info(p) for p in args.output],
    })
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[run_audit] exit_code: {returncode}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
