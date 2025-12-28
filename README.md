# Reflex AI Chat

Local chat CLI for GGUF models using `llama-cpp-python`.

## Setup

1. Create a virtual environment and install dependencies.
2. Download a GGUF model and place it in `models/`.

## Run

```powershell
python app.py
```

## Reflex UI

```powershell
pip install reflex
reflex run
```

### UI Features

- Streaming responses with typing indicator
- Enter to send
- Stop generation and reset chat
- Basic memory injection for facts like “X is a Y”
- Markdown rendering for assistant responses

## Environment Variables

- `MODEL_PATH`
- `N_CTX` (default: `2048`)
- `N_GPU_LAYERS` (default: `20`)
- `N_THREADS` (default: half of CPU cores)

## Commands

- `/exit` or `exit` or `quit`
- `/reset` clears chat history and memory
- `/memory` shows stored facts
