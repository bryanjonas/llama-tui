#!/usr/bin/env python3
"""Llama.cpp GPU Service Manager — TUI

Manages llama-server instances across 3 GPUs.
Config : ~/.llama-tui/config.json
Logs   : ~/.llama-tui/logs/gpu-N.log
Models : ~/models/  (or configured path)
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import requests
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RichLog,
    Rule,
    Static,
    Switch,
)

# ─── Paths & Constants ────────────────────────────────────────────────────────

CONFIG_DIR     = Path.home() / ".llama-tui"
CONFIG_FILE    = CONFIG_DIR / "config.json"
LOG_DIR        = CONFIG_DIR / "logs"
DOWNLOADS_DIR  = CONFIG_DIR / "downloads"
DEFAULT_MODELS = Path.home() / "models"
HF_TOKEN_FILE  = Path.home() / ".cache" / "huggingface" / "token"
SCRIPT_DIR     = Path(__file__).parent
DEFAULT_LLAMA_SERVER = str((SCRIPT_DIR / "llama-cuda" / "llama-server").resolve())

NUM_GPUS  = 3
BASE_PORT = 8080

GPU_NAMES = {
    0: "RTX 2080 Ti",
    1: "RTX 2070",
    2: "RTX 2080 Ti",
}

DEFAULT_FLAGS: dict = {
    "ctx_size":   4096,   # -c
    "gpu_layers": 99,     # -ngl
    "flash_attn": False,  # --flash-attn
    "threads":    8,      # --threads
    "parallel":   1,      # --parallel
    "mlock":      False,  # --mlock
    "no_mmap":    False,  # --no-mmap
    "cuda_visible_devices": "",  # env: CUDA_VISIBLE_DEVICES (blank -> default GPU index)
    "extra_args": "",     # free-form passthrough
}

DEFAULT_CONFIG: dict = {
    "llama_server_path": DEFAULT_LLAMA_SERVER,
    "models_dir": str(DEFAULT_MODELS),
    "base_port": BASE_PORT,
    "services": [
        {
            "gpu": i,
            "port": BASE_PORT + i,
            "model": None,
            "flags": {**DEFAULT_FLAGS, "cuda_visible_devices": str(i)},
        }
        for i in range(NUM_GPUS)
    ],
}

# ─── Config helpers ───────────────────────────────────────────────────────────

def load_config() -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_FILE.exists():
        try:
            cfg = json.loads(CONFIG_FILE.read_text())
            # Fill in any missing keys from defaults
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)
            # Migrate legacy/default server paths to local llama-cuda binary.
            configured = str(cfg.get("llama_server_path", "")).strip()
            if configured in {"", "llama-server"} or "llama-b8184/llama-server" in configured:
                cfg["llama_server_path"] = DEFAULT_LLAMA_SERVER
            # Ensure all 3 service entries exist
            existing = {s["gpu"] for s in cfg.get("services", [])}
            for i in range(NUM_GPUS):
                if i not in existing:
                    cfg["services"].append(
                        {"gpu": i, "port": BASE_PORT + i, "model": None,
                         "flags": {**DEFAULT_FLAGS, "cuda_visible_devices": str(i)}}
                    )
            # Migrate old extra_args string → structured flags dict
            for svc in cfg["services"]:
                svc_gpu = int(svc.get("gpu", 0))
                if "flags" not in svc:
                    old = svc.pop("extra_args", "")
                    svc["flags"] = {**DEFAULT_FLAGS, "cuda_visible_devices": str(svc_gpu)}
                    if old:
                        svc["flags"]["extra_args"] = old
                else:
                    # Ensure all flag keys exist
                    for k, v in DEFAULT_FLAGS.items():
                        if k == "cuda_visible_devices":
                            svc["flags"].setdefault(k, str(svc_gpu))
                        else:
                            svc["flags"].setdefault(k, v)
            cfg["services"].sort(key=lambda s: s["gpu"])
            return cfg
        except Exception:
            pass
    save_config(DEFAULT_CONFIG)
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def hf_token() -> str:
    if HF_TOKEN_FILE.exists():
        tok = HF_TOKEN_FILE.read_text().strip()
        if tok:
            return tok
    return os.environ.get("HF_TOKEN", "")


def scan_existing_llama_servers() -> dict[int, tuple[int, Optional[str], Optional[str]]]:
    """Scan /proc for running llama-server processes.

    Returns {port: (pid, model_path_or_None, cuda_visible_devices_or_None)}
    for each found instance.
    """
    import signal as _signal
    results: dict[int, tuple[int, Optional[str], Optional[str]]] = {}
    proc_dir = Path("/proc")
    for pid_dir in proc_dir.iterdir():
        if not pid_dir.name.isdigit():
            continue
        try:
            raw  = (pid_dir / "cmdline").read_bytes()
            args = raw.decode(errors="replace").split("\x00")
            if not any("llama-server" in a for a in args):
                continue
            port:  Optional[int] = None
            model: Optional[str] = None
            cuda_visible_devices: Optional[str] = None
            for i, arg in enumerate(args):
                if arg == "--port" and i + 1 < len(args):
                    try:
                        port = int(args[i + 1])
                    except ValueError:
                        pass
                elif arg == "--model" and i + 1 < len(args):
                    model = args[i + 1] or None
            try:
                raw_env = (pid_dir / "environ").read_bytes().decode(errors="replace")
                for entry in raw_env.split("\x00"):
                    if entry.startswith("CUDA_VISIBLE_DEVICES="):
                        value = entry.split("=", 1)[1].strip()
                        cuda_visible_devices = value or None
                        break
            except (PermissionError, FileNotFoundError, OSError):
                pass
            if port is not None:
                results[port] = (int(pid_dir.name), model, cuda_visible_devices)
        except (PermissionError, FileNotFoundError, OSError):
            continue
    return results


# ─── Service process management ───────────────────────────────────────────────

class ServiceProcess:
    """Manages a single llama-server subprocess bound to one GPU."""

    def __init__(self, gpu: int, port: int, flags: Optional[dict] = None):
        self.gpu    = gpu
        self.port   = port
        self.flags: dict = flags if flags is not None else dict(DEFAULT_FLAGS)
        self.model: Optional[str] = None
        self._proc:       Optional[asyncio.subprocess.Process] = None
        self._log_fh      = None
        self._external_pid: Optional[int] = None  # PID of a pre-existing process

    def _build_args(self) -> list[str]:
        """Build CLI arg list from the structured flags dict."""
        f    = self.flags
        args: list[str] = []
        ctx = int(f.get("ctx_size", 0) or 0)
        if ctx > 0:
            args += ["-c", str(ctx)]
        ngl = f.get("gpu_layers")
        if ngl is not None and int(ngl) >= 0:
            args += ["-ngl", str(int(ngl))]
        if f.get("flash_attn"):
            args += ["--flash-attn", "true"]
        thr = int(f.get("threads", 0) or 0)
        if thr > 0:
            args += ["--threads", str(thr)]
        par = int(f.get("parallel", 0) or 0)
        if par > 0:
            args += ["--parallel", str(par)]
        if f.get("mlock"):
            args.append("--mlock")
        if f.get("no_mmap"):
            args.append("--no-mmap")
        extra = (f.get("extra_args") or "").strip()
        if extra:
            # Keep TUI-managed flags authoritative; ignore duplicates in extra_args.
            managed_with_value = {"-c", "--ctx-size", "-ngl", "--threads", "--parallel"}
            managed_switches = {"--flash-attn", "--mlock", "--no-mmap"}
            tokens = shlex.split(extra)
            filtered: list[str] = []
            i = 0
            while i < len(tokens):
                t = tokens[i]
                if t in managed_with_value:
                    i += 2
                    continue
                if t in managed_switches:
                    i += 1
                    if i < len(tokens) and not tokens[i].startswith("-"):
                        i += 1
                    continue
                filtered.append(t)
                i += 1
            args += filtered
        return args

    def attach(self, pid: int, model: Optional[str] = None) -> None:
        """Attach to a llama-server that was started outside the TUI."""
        self._external_pid = pid
        if model:
            self.model = model

    @property
    def is_running(self) -> bool:
        # Process we spawned ourselves
        if self._proc is not None and self._proc.poll() is None:
            return True
        # Process we attached to — verify it's still alive
        if self._external_pid is not None:
            try:
                os.kill(self._external_pid, 0)
                return True
            except (ProcessLookupError, PermissionError):
                self._external_pid = None
        return False

    @property
    def pid(self) -> Optional[int]:
        if self._proc:
            return self._proc.pid
        return self._external_pid

    @property
    def log_path(self) -> Path:
        return LOG_DIR / f"gpu-{self.gpu}.log"

    def effective_cuda_visible_devices(self) -> str:
        value = str(self.flags.get("cuda_visible_devices", "")).strip()
        return value if value else str(self.gpu)

    async def start(self, server_bin: str, model: str) -> None:
        if self.is_running:
            await self.stop()

        self.model = model
        model_path = str(Path(model).expanduser().resolve())

        cmd  = [
            server_bin,
            "--host", "0.0.0.0",
            "--port", str(self.port),
            "--model", model_path,
        ] + self._build_args()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self.effective_cuda_visible_devices()

        # Append to log file
        log_path = self.log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(log_path, "a")
        sep = "=" * 60
        self._log_fh.write(
            f"\n{sep}\n"
            f"[{datetime.now().isoformat()}] START  gpu={self.gpu}  port={self.port}\n"
            f"ENV: CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n"
            f"CMD: {' '.join(cmd)}\n"
            f"{sep}\n"
        )
        self._log_fh.flush()

        self._proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self._log_fh,
            stderr=self._log_fh,
            start_new_session=True,
            close_fds=True,
        )

    async def stop(self) -> None:
        import signal as _signal
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            loop = asyncio.get_event_loop()
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, self._proc.wait),
                    timeout=6.0,
                )
            except asyncio.TimeoutError:
                self._proc.kill()
                await loop.run_in_executor(None, self._proc.wait)
        elif self._external_pid is not None:
            try:
                os.kill(self._external_pid, _signal.SIGTERM)
                for _ in range(60):          # wait up to 6 s
                    await asyncio.sleep(0.1)
                    try:
                        os.kill(self._external_pid, 0)
                    except ProcessLookupError:
                        break
                else:
                    os.kill(self._external_pid, _signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._external_pid = None
        if self._log_fh:
            try:
                self._log_fh.write(
                    f"[{datetime.now().isoformat()}] STOPPED gpu={self.gpu}\n"
                )
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None
        self._proc = None

    def status_str(self) -> str:
        if self.is_running:
            return f"● RUNNING  (PID {self.pid})"
        return "○ STOPPED"

    def model_str(self) -> str:
        if self.model:
            return Path(self.model).name
        return "—"


# ─── Custom Messages ──────────────────────────────────────────────────────────

class DoStartStop(Message):
    def __init__(self, gpu: int) -> None:
        self.gpu = gpu
        super().__init__()


class DoChangeModel(Message):
    def __init__(self, gpu: int) -> None:
        self.gpu = gpu
        super().__init__()


class DoViewLogs(Message):
    def __init__(self, gpu: int) -> None:
        self.gpu = gpu
        super().__init__()


class DoConfigFlags(Message):
    def __init__(self, gpu: int) -> None:
        self.gpu = gpu
        super().__init__()


# ─── Log Viewer Screen ────────────────────────────────────────────────────────

class LogScreen(Screen):
    BINDINGS = [
        Binding("escape", "go_back", "Back", show=True),
        Binding("q",      "go_back", "Back", show=False),
        Binding("c",      "clear",   "Clear log", show=True),
    ]

    def __init__(self, gpu: int) -> None:
        self.gpu = gpu
        self._tail_task: Optional[asyncio.Task] = None
        super().__init__()

    def compose(self) -> ComposeResult:
        name = GPU_NAMES[gpu := self.gpu]
        port = BASE_PORT + gpu
        yield Header(show_clock=True)
        yield Static(
            f" GPU {gpu} — {name}  |  Port {port}  |  Live Log ",
            id="log-banner",
        )
        yield RichLog(id="log-out", highlight=True, markup=False, wrap=True)
        yield Footer()

    def on_mount(self) -> None:
        self._tail_task = asyncio.create_task(self._tail())

    def on_unmount(self) -> None:
        if self._tail_task:
            self._tail_task.cancel()

    async def _tail(self) -> None:
        log  = self.query_one("#log-out", RichLog)
        path = LOG_DIR / f"gpu-{self.gpu}.log"

        if not path.exists():
            log.write("[dim]No log file yet — start the service to see output.[/dim]")
            # Wait until the file appears
            while not path.exists():
                await asyncio.sleep(1)

        # Show last 300 lines
        try:
            lines = path.read_text(errors="replace").splitlines()
            for line in lines[-300:]:
                log.write(line)
        except Exception as e:
            log.write(f"[red]Error reading log: {e}[/red]")
            return

        # Tail new content
        try:
            with open(path, "r", errors="replace") as fh:
                fh.seek(0, 2)
                while True:
                    line = fh.readline()
                    if line:
                        log.write(line.rstrip())
                    else:
                        await asyncio.sleep(0.15)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.write(f"[red]Tail error: {e}[/red]")

    def action_go_back(self) -> None:
        self.app.pop_screen()

    def action_clear(self) -> None:
        path = LOG_DIR / f"gpu-{self.gpu}.log"
        try:
            path.write_text("")
            self.query_one("#log-out", RichLog).clear()
            self.notify("Log cleared")
        except Exception as e:
            self.notify(f"Clear failed: {e}", severity="error")


# ─── Model Picker Modal ───────────────────────────────────────────────────────

class ModelPickerScreen(ModalScreen):
    """Pick a .gguf file from the models directory."""

    CSS = """
    ModelPickerScreen {
        align: center middle;
    }
    #mp-box {
        background: $surface;
        border: thick $primary;
        width: 72;
        height: 32;
        padding: 1 2;
    }
    #mp-title  { text-style: bold; margin-bottom: 1; color: $accent; }
    #mp-list   { height: 22; border: solid $primary; }
    #mp-btns   { margin-top: 1; }
    #mp-btns Button { margin-right: 1; }
    """

    def __init__(self, gpu: int, models_dir: str) -> None:
        self.gpu        = gpu
        self.models_dir = Path(models_dir).expanduser()
        super().__init__()

    def compose(self) -> ComposeResult:
        models = sorted(self.models_dir.rglob("*.gguf"))
        with Vertical(id="mp-box"):
            yield Label(
                f"Select model for GPU {self.gpu} — {GPU_NAMES[self.gpu]}",
                id="mp-title",
            )
            with ListView(id="mp-list"):
                if not models:
                    yield ListItem(
                        Label(
                            f"[dim]No .gguf files found in {self.models_dir}[/dim]"
                        )
                    )
                else:
                    for m in models:
                        yield ListItem(Label(m.name), name=str(m))
            with Horizontal(id="mp-btns"):
                yield Button("Select", variant="primary", id="mp-select")
                yield Button("Cancel", variant="default",  id="mp-cancel")

    @on(Button.Pressed, "#mp-cancel")
    def cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#mp-select")
    def select_highlighted(self) -> None:
        lv = self.query_one("#mp-list", ListView)
        if lv.highlighted_child and lv.highlighted_child.name:
            self.dismiss(lv.highlighted_child.name)
        else:
            self.notify("Highlight a model first", severity="warning")

    @on(ListView.Selected)
    def list_dbl(self, event: ListView.Selected) -> None:
        if event.item.name:
            self.dismiss(event.item.name)


# ─── HuggingFace Download Screen ─────────────────────────────────────────────

class DownloadScreen(Screen):
    BINDINGS = [
        Binding("escape", "go_back", "Back", show=True),
        Binding("q",      "go_back", "Back", show=False),
    ]

    def __init__(self) -> None:
        self._tail_task: Optional[asyncio.Task] = None
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(" HuggingFace Model Download ", id="dl-banner")
        with Vertical(id="dl-body"):
            yield Label("Repo  (org/name)  or full  https://huggingface.co/…  URL:")
            yield Input(
                placeholder="e.g.  TheBloke/Mistral-7B-v0.1-GGUF",
                id="dl-repo",
            )
            yield Label("Filename  (leave blank to list available .gguf files):")
            yield Input(
                placeholder="e.g.  mistral-7b-v0.1.Q4_K_M.gguf",
                id="dl-file",
            )
            with Horizontal(id="dl-btns"):
                yield Button("Download / List", variant="primary", id="dl-go")
                yield Button("Back", id="dl-back")
            yield RichLog(id="dl-log", highlight=True, markup=True, wrap=True)
        yield Footer()

    def on_unmount(self) -> None:
        if self._tail_task:
            self._tail_task.cancel()

    @on(Button.Pressed, "#dl-back")
    def go_back_btn(self) -> None:
        self.action_go_back()

    def action_go_back(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#dl-go")
    def start_dl(self) -> None:
        repo_val = self.query_one("#dl-repo", Input).value.strip()
        file_val = self.query_one("#dl-file", Input).value.strip()

        if not repo_val:
            self.notify("Enter a repo or URL first", severity="warning")
            return

        log = self.query_one("#dl-log", RichLog)
        log.clear()

        # ── Resolve URL and filename ──────────────────────────────────────────
        url: Optional[str] = None
        filename = file_val

        if repo_val.startswith("http"):
            url = repo_val
            if not filename:
                filename = url.split("?")[0].rstrip("/").split("/")[-1]
        else:
            repo = repo_val.strip("/")
            if not filename:
                # List mode — quick API call, no background process needed
                self._list_repo_worker(repo, log)
                return
            url = f"https://huggingface.co/{repo}/resolve/main/{filename}"

        if not filename:
            filename = url.split("?")[0].rstrip("/").split("/")[-1]

        # ── Spawn detached background downloader ──────────────────────────────
        models_dir = Path(self.app.config["models_dir"]).expanduser()
        models_dir.mkdir(parents=True, exist_ok=True)
        DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

        out_path = models_dir / filename
        log_path = DOWNLOADS_DIR / f"{filename}.log"
        log_path.write_text("")  # truncate/create before spawning

        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "downloader.py"),
            "--url",    url,
            "--output", str(out_path),
        ]

        try:
            subprocess.Popen(
                cmd,
                stdout=open(log_path, "w"),
                stderr=subprocess.STDOUT,
                start_new_session=True,  # detach — survives TUI exit
                close_fds=True,
            )
        except Exception as e:
            log.write(f"[red]Failed to spawn downloader: {e}[/red]")
            return

        log.write(f"[green]Download started as background process[/green]")
        log.write(f"[dim]Output : {out_path}[/dim]")
        log.write(f"[dim]Log    : {log_path}[/dim]")
        log.write(f"[dim]Closing the TUI will not interrupt the download.[/dim]")
        log.write("")

        # Cancel any previous tail and start a new one
        if self._tail_task:
            self._tail_task.cancel()
        self._tail_task = asyncio.create_task(self._tail_log(log, log_path))

    @work(thread=True)
    def _list_repo_worker(self, repo: str, log: RichLog) -> None:
        def emit(msg: str) -> None:
            self.app.call_from_thread(log.write, msg)

        token   = hf_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        emit(f"[yellow]Querying HuggingFace API for[/yellow] [cyan]{repo}[/cyan]…")
        try:
            resp = requests.get(
                f"https://huggingface.co/api/models/{repo}",
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            data  = resp.json()
            files = [s["rfilename"] for s in data.get("siblings", [])]
            gguf  = [f for f in files if f.endswith(".gguf")]

            if gguf:
                emit(f"[green]Found {len(gguf)} GGUF file(s) in {repo}:[/green]")
                for f in gguf:
                    emit(f"  [cyan]{f}[/cyan]")
                emit("\n[dim]Enter one of the filenames above and click Download.[/dim]")
            else:
                emit(f"[yellow]No .gguf files in {repo}.[/yellow]")
                if files:
                    emit("Files present: " + ", ".join(files[:30]))
        except Exception as e:
            emit(f"[red]API error: {e}[/red]")

    async def _tail_log(self, log: RichLog, log_path: Path) -> None:
        """Tail the download log file written by the background process."""
        while not log_path.exists():
            await asyncio.sleep(0.2)
        try:
            with open(log_path, "r", errors="replace") as fh:
                while True:
                    line = fh.readline()
                    if line:
                        log.write(line.rstrip())
                    else:
                        await asyncio.sleep(0.3)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.write(f"[red]Tail error: {e}[/red]")


# ─── Service Flags Modal ─────────────────────────────────────────────────────

class ServiceFlagsScreen(ModalScreen):
    """Per-GPU optimization flags editor."""

    CSS = """
    ServiceFlagsScreen {
        align: center middle;
    }
    #flags-box {
        background: $surface;
        border: thick $primary;
        width: 64;
        height: auto;
        padding: 1 2;
    }
    #flags-title { text-style: bold; margin-bottom: 1; color: $accent; }
    .flags-row { height: 3; margin-bottom: 0; }
    .flags-lbl  { width: 22; content-align: left middle; }
    .flags-val  { width: 1fr; }
    #f-extra-lbl { margin-top: 1; }
    #flags-btns { margin-top: 2; height: 3; }
    #flags-btns Button { margin-right: 1; }
    """

    def __init__(self, gpu: int, flags: dict) -> None:
        self.gpu   = gpu
        self._flags = dict(flags)
        super().__init__()

    def compose(self) -> ComposeResult:
        f = self._flags
        with Vertical(id="flags-box"):
            yield Label(
                f"Flags — GPU {self.gpu} ({GPU_NAMES[self.gpu]})",
                id="flags-title",
            )
            with Horizontal(classes="flags-row"):
                yield Label("-c  context size:", classes="flags-lbl")
                yield Input(value=str(f.get("ctx_size", 4096)), id="f-ctx", classes="flags-val")
            with Horizontal(classes="flags-row"):
                yield Label("-ngl  GPU layers:", classes="flags-lbl")
                yield Input(value=str(f.get("gpu_layers", 99)), id="f-ngl", classes="flags-val")
            with Horizontal(classes="flags-row"):
                yield Label("--threads:", classes="flags-lbl")
                yield Input(value=str(f.get("threads", 8)), id="f-thr", classes="flags-val")
            with Horizontal(classes="flags-row"):
                yield Label("--parallel:", classes="flags-lbl")
                yield Input(value=str(f.get("parallel", 1)), id="f-par", classes="flags-val")
            with Horizontal(classes="flags-row"):
                yield Label("--flash-attn:", classes="flags-lbl")
                yield Switch(value=bool(f.get("flash_attn", False)), id="f-fa")
            with Horizontal(classes="flags-row"):
                yield Label("--mlock:", classes="flags-lbl")
                yield Switch(value=bool(f.get("mlock", False)), id="f-ml")
            with Horizontal(classes="flags-row"):
                yield Label("--no-mmap:", classes="flags-lbl")
                yield Switch(value=bool(f.get("no_mmap", False)), id="f-nm")
            with Horizontal(classes="flags-row"):
                yield Label("CUDA_VISIBLE_DEVICES:", classes="flags-lbl")
                yield Input(
                    value=str(f.get("cuda_visible_devices", str(self.gpu))),
                    id="f-cvd",
                    classes="flags-val",
                )
            yield Label("Extra args (free-form):", id="f-extra-lbl")
            yield Input(value=f.get("extra_args", ""), id="f-extra")
            with Horizontal(id="flags-btns"):
                yield Button("Apply", variant="primary", id="f-apply")
                yield Button("Cancel", variant="default",  id="f-cancel")

    @on(Button.Pressed, "#f-cancel")
    def cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#f-apply")
    def apply(self) -> None:
        def _int(widget_id: str, default: int) -> int:
            try:
                return int(self.query_one(widget_id, Input).value.strip())
            except (ValueError, TypeError):
                return default

        result = {
            "ctx_size":   _int("#f-ctx", 4096),
            "gpu_layers": _int("#f-ngl", 99),
            "threads":    _int("#f-thr", 8),
            "parallel":   _int("#f-par", 1),
            "flash_attn": self.query_one("#f-fa", Switch).value,
            "mlock":      self.query_one("#f-ml", Switch).value,
            "no_mmap":    self.query_one("#f-nm", Switch).value,
            "cuda_visible_devices": self.query_one("#f-cvd", Input).value.strip(),
            "extra_args": self.query_one("#f-extra", Input).value.strip(),
        }
        self.dismiss(result)


# ─── Settings Screen ──────────────────────────────────────────────────────────

class SettingsScreen(Screen):
    BINDINGS = [
        Binding("escape", "go_back", "Back", show=True),
    ]

    def compose(self) -> ComposeResult:
        cfg = self.app.config
        yield Header(show_clock=True)
        yield Static(" Settings ", id="set-banner")
        with Vertical(id="set-body"):
            yield Label("llama-server binary path  (full path or name if in PATH):")
            yield Input(value=cfg.get("llama_server_path", DEFAULT_LLAMA_SERVER), id="s-bin")
            yield Label("Models directory:")
            yield Input(value=cfg.get("models_dir", str(DEFAULT_MODELS)), id="s-models")
            yield Label(
                "[dim]Per-GPU optimization flags are edited via the ⚙ Flags button on each panel.[/dim]",
                classes="svc-label",
            )
            with Horizontal(id="set-btns"):
                yield Button("Save", variant="primary", id="s-save")
                yield Button("Cancel", id="s-cancel")
        yield Footer()

    @on(Button.Pressed, "#s-save")
    def save(self) -> None:
        cfg = self.app.config
        cfg["llama_server_path"] = self.query_one("#s-bin",    Input).value.strip()
        cfg["models_dir"]        = self.query_one("#s-models", Input).value.strip()
        save_config(cfg)
        self.notify("Settings saved")
        self.app.pop_screen()

    @on(Button.Pressed, "#s-cancel")
    def cancel(self) -> None:
        self.action_go_back()

    def action_go_back(self) -> None:
        self.app.pop_screen()


# ─── Service Panel Widget ─────────────────────────────────────────────────────

class ServicePanel(Static):
    """One GPU's control panel."""

    def __init__(self, gpu: int) -> None:
        self.gpu = gpu
        super().__init__(id=f"panel-{gpu}")

    def compose(self) -> ComposeResult:
        gpu  = self.gpu
        name = GPU_NAMES[gpu]
        port = BASE_PORT + gpu

        with Vertical(classes="pnl-v"):
            # Header row
            yield Static(f"GPU {gpu}", classes="pnl-gpu-idx")
            yield Static(name,         classes="pnl-gpu-name")
            yield Static(f":{port}",   classes="pnl-port")
            yield Rule()

            # Dynamic status
            yield Static("○ STOPPED", id=f"svc-status-{gpu}", classes="pnl-status stopped")
            yield Static("—",         id=f"svc-model-{gpu}",  classes="pnl-model")
            yield Rule()

            # Action buttons
            yield Button("▶  Start",        id=f"btn-ss-{gpu}",  variant="success", classes="pnl-btn")
            yield Button("⊞  Change Model",  id=f"btn-mdl-{gpu}", variant="default",  classes="pnl-btn")
            yield Button("⚙  Flags",         id=f"btn-flg-{gpu}", variant="default",  classes="pnl-btn")
            yield Button("≡  View Logs",     id=f"btn-log-{gpu}", variant="default",  classes="pnl-btn")

    @on(Button.Pressed)
    def btn_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == f"btn-ss-{self.gpu}":
            self.post_message(DoStartStop(self.gpu))
        elif bid == f"btn-mdl-{self.gpu}":
            self.post_message(DoChangeModel(self.gpu))
        elif bid == f"btn-flg-{self.gpu}":
            self.post_message(DoConfigFlags(self.gpu))
        elif bid == f"btn-log-{self.gpu}":
            self.post_message(DoViewLogs(self.gpu))

    def refresh_state(self, svc: ServiceProcess) -> None:
        status_w = self.query_one(f"#svc-status-{self.gpu}", Static)
        model_w  = self.query_one(f"#svc-model-{self.gpu}",  Static)
        ss_btn   = self.query_one(f"#btn-ss-{self.gpu}",     Button)

        if svc.is_running:
            status_w.update(f"● RUNNING  (PID {svc.pid})")
            status_w.set_classes("pnl-status running")
            ss_btn.label   = "■  Stop"
            ss_btn.variant = "error"
        else:
            status_w.update("○ STOPPED")
            status_w.set_classes("pnl-status stopped")
            ss_btn.label   = "▶  Start"
            ss_btn.variant = "success"

        model_w.update(svc.model_str())


# ─── Main App ─────────────────────────────────────────────────────────────────

class LlamaManager(App):
    TITLE   = "Llama.cpp  GPU  Service  Manager"
    CSS_PATH = None

    CSS = """
    /* ── App chrome ── */
    Header { background: $primary-darken-2; }
    Footer { background: $primary-darken-2; }

    /* ── Main grid ── */
    #grid {
        layout: horizontal;
        height: 1fr;
        padding: 0 1;
    }

    /* ── Service panels ── */
    ServicePanel {
        width: 1fr;
        height: 100%;
        border: round $primary-darken-1;
        margin: 1;
        padding: 0 1;
    }
    .pnl-v       { height: 100%; }
    .pnl-gpu-idx { text-style: bold; color: $accent; margin-top: 1; }
    .pnl-gpu-name{ color: $text-muted; }
    .pnl-port    { color: $text-muted; }
    .pnl-status  { text-style: bold; margin-top: 1; }
    .running     { color: $success; }
    .stopped     { color: $text-disabled; }
    .pnl-model   { color: $text-muted; margin-bottom: 1; overflow: hidden; }
    .pnl-btn     { width: 100%; margin-bottom: 1; }

    /* ── Log screen ── */
    #log-banner  { background: $primary; color: $text; padding: 0 2; text-style: bold; }
    #log-out     { height: 1fr; }

    /* ── Download screen ── */
    #dl-banner   { background: $success-darken-2; color: $text; padding: 0 2; text-style: bold; }
    #dl-body     { padding: 1 2; height: 1fr; }
    #dl-body Label { margin-top: 1; }
    #dl-body Input { margin-bottom: 1; }
    #dl-btns     { margin-bottom: 1; }
    #dl-btns Button { margin-right: 1; }
    #dl-log      { height: 1fr; border: solid $primary; }

    /* ── Settings screen ── */
    #set-banner  { background: $warning-darken-2; color: $text; padding: 0 2; text-style: bold; }
    #set-body    { padding: 1 2; }
    #set-body Label { margin-top: 1; }
    #set-body Input { margin-bottom: 1; }
    .svc-label   { color: $text-muted; margin-top: 1; }
    #set-btns    { margin-top: 2; }
    #set-btns Button { margin-right: 1; }
    """

    BINDINGS = [
        Binding("d",       "download",   "Download Model"),
        Binding("s",       "settings",   "Settings"),
        Binding("r",       "refresh",    "Refresh"),
        Binding("q",       "quit",       "Quit (keep services)"),
        Binding("Q",       "quit_stop",  "Quit + stop all"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.config = load_config()
        # Build ServiceProcess objects from config
        self.services: dict[int, ServiceProcess] = {}
        for svc_cfg in self.config["services"]:
            gpu   = svc_cfg["gpu"]
            port  = svc_cfg["port"]
            flags = svc_cfg.get("flags", dict(DEFAULT_FLAGS))
            sp    = ServiceProcess(gpu, port, flags)
            sp.model = svc_cfg.get("model")  # restore last model
            self.services[gpu] = sp

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="grid"):
            for i in range(NUM_GPUS):
                yield ServicePanel(i)
        yield Footer()

    def on_mount(self) -> None:
        # Attach to any llama-server processes already running
        found = scan_existing_llama_servers()
        for gpu, svc in self.services.items():
            if svc.port in found:
                pid, model, cuda_visible_devices = found[svc.port]
                svc.attach(pid, model)   # model from cmdline takes priority
                cuda = cuda_visible_devices if cuda_visible_devices else "<unset>"
                self.notify(
                    f"GPU {gpu}: attached to existing PID {pid} (CUDA_VISIBLE_DEVICES={cuda})",
                    severity="information",
                )
                if cuda_visible_devices is None:
                    self.notify(
                        f"GPU {gpu}: PID {pid} has no CUDA_VISIBLE_DEVICES; it may span multiple GPUs.",
                        severity="warning",
                    )
                elif "," in cuda_visible_devices:
                    self.notify(
                        f"GPU {gpu}: PID {pid} exposes multiple GPUs ({cuda_visible_devices}).",
                        severity="warning",
                    )
                elif cuda_visible_devices != svc.effective_cuda_visible_devices():
                    self.notify(
                        (
                            f"GPU {gpu}: PID {pid} is pinned to "
                            f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}, "
                            f"expected {svc.effective_cuda_visible_devices()}."
                        ),
                        severity="warning",
                    )

        for i in range(NUM_GPUS):
            self._refresh_panel(i)
        # Poll process status every 2 s
        self.set_interval(2.0, self._poll_services)

    # ── Panel refresh ─────────────────────────────────────────────────────────

    def _refresh_panel(self, gpu: int) -> None:
        try:
            panel = self.query_one(f"#panel-{gpu}", ServicePanel)
            panel.refresh_state(self.services[gpu])
        except Exception:
            pass

    def _poll_services(self) -> None:
        for i in range(NUM_GPUS):
            self._refresh_panel(i)

    # ── Message handlers ──────────────────────────────────────────────────────

    @on(DoStartStop)
    def on_start_stop(self, msg: DoStartStop) -> None:
        svc = self.services[msg.gpu]
        if svc.is_running:
            self._stop_svc(msg.gpu)
        else:
            self._start_svc(msg.gpu)

    @on(DoChangeModel)
    def on_change_model(self, msg: DoChangeModel) -> None:
        gpu = msg.gpu
        models_dir = self.config.get("models_dir", str(DEFAULT_MODELS))

        def picked(model_path: Optional[str]) -> None:
            if model_path:
                self._apply_model(gpu, model_path)

        self.push_screen(ModelPickerScreen(gpu, models_dir), picked)

    @on(DoViewLogs)
    def on_view_logs(self, msg: DoViewLogs) -> None:
        self.push_screen(LogScreen(msg.gpu))

    @on(DoConfigFlags)
    def on_config_flags(self, msg: DoConfigFlags) -> None:
        gpu = msg.gpu
        svc = self.services[gpu]

        def applied(new_flags: Optional[dict]) -> None:
            if new_flags is None:
                return
            svc.flags = new_flags
            for s in self.config["services"]:
                if s["gpu"] == gpu:
                    s["flags"] = new_flags
            save_config(self.config)
            self.notify(f"GPU {gpu}: Flags saved")

        self.push_screen(ServiceFlagsScreen(gpu, svc.flags), applied)

    # ── Service control (workers) ─────────────────────────────────────────────

    @work
    async def _start_svc(self, gpu: int) -> None:
        svc = self.services[gpu]
        if not svc.model:
            self.notify(
                f"GPU {gpu}: No model selected — use ⊞ Change Model first.",
                severity="warning",
            )
            return
        server = self.config.get("llama_server_path", DEFAULT_LLAMA_SERVER)
        self.notify(f"GPU {gpu}: Starting on port {svc.port}…")
        try:
            await svc.start(server, svc.model)
            self._refresh_panel(gpu)
            self.notify(f"GPU {gpu}: Running  (PID {svc.pid})", severity="information")
        except FileNotFoundError:
            self.notify(
                f"GPU {gpu}: '{server}' not found — check Settings.",
                severity="error",
            )
        except Exception as e:
            self.notify(f"GPU {gpu}: Start failed — {e}", severity="error")

    @work
    async def _stop_svc(self, gpu: int) -> None:
        svc = self.services[gpu]
        self.notify(f"GPU {gpu}: Stopping…")
        await svc.stop()
        self._refresh_panel(gpu)
        self.notify(f"GPU {gpu}: Stopped")

    @work
    async def _apply_model(self, gpu: int, model: str) -> None:
        svc       = self.services[gpu]
        svc.model = model

        # Persist to config
        for s in self.config["services"]:
            if s["gpu"] == gpu:
                s["model"] = model
        save_config(self.config)
        self._refresh_panel(gpu)
        self.notify(f"GPU {gpu}: Model → {Path(model).name}")

        if svc.is_running:
            self.notify(f"GPU {gpu}: Restarting with new model…")
            await svc.stop()
            self._refresh_panel(gpu)
            await asyncio.sleep(0.5)
            server = self.config.get("llama_server_path", DEFAULT_LLAMA_SERVER)
            try:
                await svc.start(server, model)
                self._refresh_panel(gpu)
                self.notify(f"GPU {gpu}: Restarted  (PID {svc.pid})", severity="information")
            except Exception as e:
                self.notify(f"GPU {gpu}: Restart failed — {e}", severity="error")

    # ── App-level actions ─────────────────────────────────────────────────────

    def action_download(self) -> None:
        self.push_screen(DownloadScreen())

    def action_settings(self) -> None:
        self.push_screen(SettingsScreen())

    def action_refresh(self) -> None:
        self._poll_services()

    def action_quit(self) -> None:
        """Exit the TUI; leave all services running."""
        for svc in self.services.values():
            if svc._log_fh:
                try:
                    svc._log_fh.close()
                except Exception:
                    pass
                svc._log_fh = None
            svc._proc = None  # drop reference; process keeps running
        self.exit()

    async def action_quit_stop(self) -> None:
        """Stop all services then exit."""
        for g, svc in self.services.items():
            if svc.is_running:
                await svc.stop()
        self.exit()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    DEFAULT_MODELS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LlamaManager().run()


if __name__ == "__main__":
    main()
