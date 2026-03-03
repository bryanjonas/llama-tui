# CLAUDE.md — llama-tui

Project context and gotchas for AI-assisted development.

## What this project is

A single-file Textual TUI (`app.py`) that manages `llama-server` (llama.cpp) instances across 3 GPUs. One `ServiceProcess` object per GPU, one `ServicePanel` widget per GPU. Everything is in `app.py`; `downloader.py` is a standalone script invoked as a detached subprocess.

## Environment (development machine)

- GPUs: 0 = RTX 2080 Ti, 1 = RTX 2070, 2 = RTX 2080 Ti
- Python venv: `.venv/` (textual 8.x, requests)
- Models dir: `~/models/` (GGUF files)
- HF token: `~/.cache/huggingface/token`
- Runtime config: `~/.llama-tui/config.json`
- Server logs: `~/.llama-tui/logs/gpu-N.log`
- Download logs: `~/.llama-tui/downloads/<filename>.log`

## How to run

```bash
./run.sh        # auto-creates .venv on first run
# or
.venv/bin/python app.py
```

## Architecture notes

### ServiceProcess lifecycle
Servers are started with `subprocess.Popen(..., start_new_session=True, close_fds=True)` — **not** `asyncio.create_subprocess_exec`. This is intentional: asyncio subprocess transports call `terminate()` on their child when the event loop shuts down, which would kill servers on TUI exit. `subprocess.Popen` is untracked by asyncio and survives the TUI process exiting.

`action_quit()` (`q`) sets `svc._proc = None` before calling `self.exit()`, dropping the Python reference without touching the OS process. `action_quit_stop()` (`Q`) explicitly calls `svc.stop()` which sends SIGTERM.

`stop()` uses `loop.run_in_executor(None, self._proc.wait)` because `Popen.wait()` is blocking.

### RichLog and server log content
The log viewer (`LogScreen`) uses `RichLog(markup=False)`. This is intentional — llama-server output contains tokens like `[/INST]`, `[INST]`, `</s>` that Rich's markup parser would crash on. The download screen's `RichLog` keeps `markup=True` because the app writes Rich markup to it directly.

### Flash attention flag
The llama-server build in use requires `--flash-attn true` (key=value form), not the bare `--flash-attn` flag. `_build_args()` reflects this.

### External process attach
On `on_mount`, the app scans `/proc` for running `llama-server` processes and calls `svc.attach(pid, model)` if a matching port is found. Attached external processes are tracked via `_external_pid` (not `_proc`). Pressing `q` does not stop external processes.

## Key files

| File | Purpose |
|------|---------|
| `app.py` | Entire TUI application (single file) |
| `downloader.py` | Detached HF downloader, spawned by `DownloadScreen` |
| `run.sh` | Launcher — creates `.venv` if missing, then runs `app.py` |
| `requirements.txt` | `textual>=0.70.0`, `requests>=2.28.0` |

## Ports

| GPU | Port |
|-----|------|
| 0 (RTX 2080 Ti) | 8080 |
| 1 (RTX 2070)    | 8081 |
| 2 (RTX 2080 Ti) | 8082 |
