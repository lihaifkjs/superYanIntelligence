#!/usr/bin/env python3
"""Download a model with directory-size based progress monitoring.

Supports Hugging Face (`hf download`) and git-lfs (`git lfs pull`) backends.
Prints / logs timestamp, downloaded GB, percent, speed and ETA so that the
background task can be observed without relying only on the CLI progress bar.
"""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


def get_dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            total += get_dir_size(Path(entry.path))
        elif entry.is_file(follow_symlinks=False):
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


def human_gb(nbytes: int) -> str:
    return f"{nbytes / 1024 ** 3:.2f} GB"


def human_mb_s(bps: float) -> str:
    return f"{bps / 1024 ** 2:.2f} MB/s"


def monitor(local_dir: Path, total_bytes: int, interval: int, log_file: Path, stop_event: threading.Event):
    prev_bytes = None
    prev_time = None
    with open(log_file, "a", encoding="utf-8") as f:
        while not stop_event.is_set():
            now = time.time()
            cur_bytes = get_dir_size(local_dir)
            pct = cur_bytes / total_bytes * 100 if total_bytes else 0
            line_parts = [
                datetime.now().isoformat(timespec="seconds"),
                f"downloaded={human_gb(cur_bytes)}",
                f"total={human_gb(total_bytes)}",
                f"percent={pct:.1f}%",
            ]
            if prev_bytes is not None and prev_time is not None and now > prev_time:
                delta_bytes = cur_bytes - prev_bytes
                delta_t = now - prev_time
                rate = delta_bytes / delta_t if delta_t > 0 else 0
                line_parts.append(f"speed={human_mb_s(rate)}")
                if rate > 0 and cur_bytes < total_bytes:
                    eta_s = (total_bytes - cur_bytes) / rate
                    eta_m = eta_s / 60
                    line_parts.append(f"eta={eta_m:.1f} min")
            line = " | ".join(line_parts)
            f.write(line + "\n")
            f.flush()
            print(line, flush=True)
            prev_bytes = cur_bytes
            prev_time = now
            stop_event.wait(interval)


def run_hf_download(repo: str, local_dir: Path, max_workers: int, cli_log: Path) -> int:
    hf_exe = shutil.which("hf")
    if hf_exe is None:
        print("ERROR: hf CLI not found in PATH", file=sys.stderr)
        return 1
    cmd = [
        hf_exe,
        "download",
        repo,
        "--local-dir",
        str(local_dir),
        "--max-workers",
        str(max_workers),
    ]
    print(f"Running: {' '.join(cmd)}")
    with open(cli_log, "wb") as out:
        proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            return proc.wait()


def run_git_lfs_download(repo_url: str, local_dir: Path, cli_log: Path, lfs_progress_log: Path) -> int:
    git_exe = shutil.which("git")
    if git_exe is None:
        print("ERROR: git not found in PATH", file=sys.stderr)
        return 1

    if not (local_dir / ".git").exists():
        local_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            git_exe,
            "clone",
            "--depth",
            "1",
            "--",
            repo_url,
            str(local_dir),
        ]
        print(f"Running: {' '.join(cmd)}")
        with open(cli_log, "wb") as out:
            proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT)
            try:
                ret = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                ret = proc.wait()
            if ret != 0:
                return ret

    cmd = [git_exe, "lfs", "pull"]
    print(f"Running in {local_dir}: {' '.join(cmd)}")
    env = os.environ.copy()
    env["GIT_LFS_PROGRESS"] = str(lfs_progress_log)
    with open(cli_log, "ab") as out:
        proc = subprocess.Popen(cmd, cwd=str(local_dir), env=env, stdout=out, stderr=subprocess.STDOUT)
        try:
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            return proc.wait()


def main():
    parser = argparse.ArgumentParser(description="Download a model and monitor progress.")
    parser.add_argument("--backend", choices=["hf", "git-lfs"], default="hf", help="Download backend")
    parser.add_argument("--repo", required=True, help="HF repo id (hf) or display name (git-lfs)")
    parser.add_argument("--git-url", help="Git clone URL (required for git-lfs if local-dir is not a repo)")
    parser.add_argument("--local-dir", required=True, help="Where to place the model files")
    parser.add_argument("--total-gb", type=float, required=True, help="Expected total size in GB for ETA")
    parser.add_argument("--log-dir", default="logs", help="Directory for progress logs")
    parser.add_argument("--interval", type=int, default=30, help="Progress sampling interval in seconds")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel download workers (hf only)")
    args = parser.parse_args()

    local_dir = Path(args.local_dir).resolve()
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    local_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = int(args.total_gb * 1024 ** 3)
    progress_log = log_dir / "download_progress.log"
    cli_log = log_dir / "download_cli.log"
    lfs_progress_log = log_dir / "download_lfs_progress.log"

    print(f"[{datetime.now().isoformat(timespec='seconds')}] Start downloading {args.repo} -> {local_dir}")
    print(f"Expected total: {human_gb(total_bytes)}")
    print(f"CLI log: {cli_log}")
    print(f"Progress log: {progress_log}")

    stop_event = threading.Event()
    mon_thread = threading.Thread(
        target=monitor,
        args=(local_dir, total_bytes, args.interval, progress_log, stop_event),
        daemon=True,
    )
    mon_thread.start()

    if args.backend == "hf":
        ret = run_hf_download(args.repo, local_dir, args.max_workers, cli_log)
    elif args.backend == "git-lfs":
        if args.git_url is None and not (local_dir / ".git").exists():
            print("ERROR: --git-url is required when local-dir is not an existing git repo", file=sys.stderr)
            stop_event.set()
            mon_thread.join()
            sys.exit(1)
        ret = run_git_lfs_download(args.git_url, local_dir, cli_log, lfs_progress_log)
    else:
        ret = 1

    stop_event.set()
    mon_thread.join(timeout=args.interval + 5)

    final_bytes = get_dir_size(local_dir)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Download finished with exit code {ret}")
    print(f"Final size: {human_gb(final_bytes)} ({final_bytes / total_bytes * 100:.1f}% of expected)")
    sys.exit(ret)


if __name__ == "__main__":
    main()
