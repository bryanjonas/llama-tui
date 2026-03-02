#!/usr/bin/env python3
"""
Standalone background downloader for llama-tui.
Runs detached from the TUI — survives session drops.

Usage:
  python3 downloader.py --url URL --output PATH

HF token is read from ~/.cache/huggingface/token or $HF_TOKEN — never
passed on the command line so it won't appear in `ps aux`.

Progress is written to stdout (caller redirects to a log file).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import requests

_HF_TOKEN_FILE = Path.home() / ".cache" / "huggingface" / "token"


def _hf_token() -> str:
    if _HF_TOKEN_FILE.exists():
        tok = _HF_TOKEN_FILE.read_text().strip()
        if tok:
            return tok
    return os.environ.get("HF_TOKEN", "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",    required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        print(msg, flush=True)

    token   = _hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    log(f"[START]  {out_path.name}")
    log(f"URL  : {args.url}")
    log(f"Dest : {out_path}")
    log(f"PID  : {os.getpid()}")
    log("")

    try:
        with requests.get(args.url, headers=headers, stream=True, timeout=30) as resp:
            resp.raise_for_status()
            total      = int(resp.headers.get("Content-Length", 0))
            done       = 0
            chunk_size = 4 * 1024 * 1024   # 4 MB chunks
            last_log   = 0.0
            t_start    = time.monotonic()

            with open(out_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    fh.write(chunk)
                    done += len(chunk)
                    now  = time.monotonic()
                    if now - last_log >= 2.0:
                        last_log = now
                        elapsed  = now - t_start
                        speed    = done / elapsed / (1024 * 1024) if elapsed else 0
                        if total:
                            pct = done / total * 100
                            log(
                                f"  {done/1024/1024:7.1f} / {total/1024/1024:.1f} MB"
                                f"  ({pct:.0f}%)  {speed:.1f} MB/s"
                            )
                        else:
                            log(f"  {done/1024/1024:.1f} MB  {speed:.1f} MB/s")

        elapsed = time.monotonic() - t_start
        size_mb = out_path.stat().st_size / (1024 * 1024)
        log("")
        log(f"[DONE]  {out_path.name}  ({size_mb:.1f} MB in {elapsed:.0f}s)")
        return 0

    except requests.HTTPError as e:
        log(f"[ERROR]  HTTP {e.response.status_code}: {e}")
        # Remove partial file
        out_path.unlink(missing_ok=True)
        return 1
    except KeyboardInterrupt:
        log("[CANCELLED]")
        out_path.unlink(missing_ok=True)
        return 1
    except Exception as e:
        log(f"[ERROR]  {e}")
        out_path.unlink(missing_ok=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
