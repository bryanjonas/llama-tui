# llama-tui

A terminal UI for managing multiple [llama.cpp](https://github.com/ggml-org/llama.cpp) server instances across several GPUs simultaneously.

## Motivation

Running several `llama-server` instances across multiple GPUs is tedious to manage from the command line — each one needs a different port, the right `CUDA_VISIBLE_DEVICES`, log redirection, and manual process tracking. Switching models means killing a process, retyping a long command, and hoping you remembered the right flags.

`llama-tui` wraps all of that in a single terminal dashboard. You can see all three servers at a glance, start or stop any of them, swap models, tune per-GPU flags, and download new GGUF models from HuggingFace — without leaving the terminal or remembering a single command-line argument.

Quitting the TUI leaves every server running. The servers are not children of the TUI process and will keep serving requests until you explicitly stop them.

## Requirements

- Python 3.9+
- NVIDIA GPUs with CUDA (the app expects 3 GPUs by default; edit `NUM_GPUS` in `app.py` to change this)
- A `llama-server` binary ([pre-built releases](https://github.com/ggml-org/llama.cpp/releases))

## Installation

```bash
git clone <repo>
cd llama-tui
./run.sh          # creates .venv, installs deps, launches the app
```

`run.sh` automatically creates a Python virtualenv on first run and installs the two dependencies (`textual`, `requests`). After that it just launches the app.

You can also manage the environment manually:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

## First-time setup

1. **Set the llama-server path** — press `s` to open Settings and enter the full path to your `llama-server` binary (or just `llama-server` if it is on your `PATH`).
2. **Set the models directory** — also in Settings. Defaults to `~/models/`. Any `.gguf` files found recursively under this directory will appear in the model picker.
3. **Select a model per GPU** — press `⊞ Change Model` on a panel to pick a `.gguf` file, then press `▶ Start` to launch the server.

## Layout

The main screen shows one panel per GPU side by side:

```
┌─ GPU 0 ──────────┐  ┌─ GPU 1 ──────────┐  ┌─ GPU 2 ──────────┐
│ RTX 2080 Ti :8080│  │ RTX 2070    :8081│  │ RTX 2080 Ti :8082│
│                  │  │                  │  │                  │
│ ● RUNNING (1234) │  │ ○ STOPPED        │  │ ● RUNNING (5678) │
│ mistral-7b.gguf  │  │ —                │  │ llama-3.gguf     │
│                  │  │                  │  │                  │
│ [■ Stop        ] │  │ [▶ Start      ] │  │ [■ Stop        ] │
│ [⊞ Change Model] │  │ [⊞ Change Model] │  │ [⊞ Change Model] │
│ [⚙ Flags       ] │  │ [⚙ Flags       ] │  │ [⚙ Flags       ] │
│ [≡ View Logs   ] │  │ [≡ View Logs   ] │  │ [≡ View Logs   ] │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

Each server listens on `0.0.0.0` and runs with `CUDA_VISIBLE_DEVICES` set to its GPU index, so GPU 0 → port 8080, GPU 1 → port 8081, GPU 2 → port 8082.

## Keybindings

| Key | Action |
|-----|--------|
| `d` | Open the HuggingFace download screen |
| `s` | Open Settings |
| `r` | Refresh all panels |
| `q` | Quit the TUI — **servers keep running** |
| `Q` | Quit and stop all servers |

## Per-GPU flags

Press `⚙ Flags` on any panel to open the flags editor for that GPU:

| Flag | llama-server argument | Default |
|------|-----------------------|---------|
| Context size | `-c` | 4096 |
| GPU layers | `-ngl` | 99 |
| Threads | `--threads` | 8 |
| Parallel slots | `--parallel` | 1 |
| Flash Attention | `--flash-attn true` | off |
| mlock | `--mlock` | off |
| no-mmap | `--no-mmap` | off |
| Extra args | passed through verbatim | — |

Flags are saved to `~/.llama-tui/config.json` and applied the next time a server is started. Changing flags does not restart a running server automatically.

## Downloading models

Press `d` to open the download screen.

- **List available files** — enter a HuggingFace repo (e.g. `TheBloke/Mistral-7B-v0.1-GGUF`) and leave the filename blank, then click **Download / List**. The app queries the HF API and lists all `.gguf` files in the repo.
- **Download a file** — enter the repo and the filename, then click **Download / List**. The download runs as a fully detached background process (`downloader.py`) that survives closing the TUI. Progress is streamed into the log view.
- **Direct URL** — paste a full `https://huggingface.co/…` URL instead of a repo slug.

Downloaded files land in the configured models directory. The HuggingFace token is read from `~/.cache/huggingface/token` or the `$HF_TOKEN` environment variable and is never exposed on the command line.

## Logs

Press `≡ View Logs` on any panel to tail the live log for that GPU's server. Logs are stored at:

```
~/.llama-tui/logs/gpu-0.log
~/.llama-tui/logs/gpu-1.log
~/.llama-tui/logs/gpu-2.log
```

Press `c` inside the log viewer to clear the log file. Download progress logs are at `~/.llama-tui/downloads/<filename>.log`.

## Config file

`~/.llama-tui/config.json` is created automatically on first run. You can edit it by hand if needed:

```json
{
  "llama_server_path": "/path/to/llama-server",
  "models_dir": "/home/user/models",
  "base_port": 8080,
  "services": [
    {
      "gpu": 0,
      "port": 8080,
      "model": "/home/user/models/mistral-7b.Q4_K_M.gguf",
      "flags": {
        "ctx_size": 4096,
        "gpu_layers": 99,
        "flash_attn": false,
        "threads": 8,
        "parallel": 1,
        "mlock": false,
        "no_mmap": false,
        "extra_args": ""
      }
    }
  ]
}
```

## Attaching to existing servers

If `llama-server` processes are already running when the TUI starts, it scans `/proc` and automatically attaches to any instance whose port matches a configured service. The panel will show the PID and model name (if readable from the process command line). These pre-existing servers are not stopped when you press `q`.
