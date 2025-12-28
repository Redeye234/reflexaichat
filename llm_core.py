import os
import re
from pathlib import Path
from typing import List, Dict, TypedDict, Optional

from llama_cpp import Llama

MODEL_PATH = os.environ.get("MODEL_PATH", "models/phi-2.Q4_K_M.gguf")

# For 4GB VRAM GPU, start small:
# - n_ctx: keep 2048
# - n_gpu_layers: try 10, 20, 30... until you hit VRAM limit
N_CTX = int(os.environ.get("N_CTX", "2048"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "20"))  # set 0 for CPU-only
CPU_COUNT = os.cpu_count() or 2
N_THREADS = int(os.environ.get("N_THREADS", str(max(2, CPU_COUNT // 2))))

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer only the user's question, concisely. "
    "Do not add any extra text, examples, or unrelated content. "
    "If you don't know, say you don't know."
)

class HistoryMessage(TypedDict):
    role: str
    content: str

FACT_MEMORY: List[str] = []
FACT_PATTERN = re.compile(
    r"^\s*([A-Za-z0-9 _-]{2,80})\s+is\s+(an?|the)\s+(.{2,120})\s*$",
    re.IGNORECASE,
)

_LLM: Optional[Llama] = None

def get_llm() -> Llama:
    global _LLM
    if _LLM is None:
        path = Path(MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path.resolve()}")
        _LLM = Llama(
            model_path=str(path),
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,   # 0 => CPU-only
            n_threads=N_THREADS,
            verbose=False,
        )
    return _LLM

def maybe_store_fact(text: str) -> None:
    match = FACT_PATTERN.match(text)
    if not match:
        return
    subject = match.group(1).strip()
    predicate = match.group(3).strip().rstrip(".")
    fact = f"{subject} is {predicate}"
    if fact not in FACT_MEMORY:
        FACT_MEMORY.append(fact)

def reset_memory() -> None:
    FACT_MEMORY.clear()

def get_memory() -> List[str]:
    return list(FACT_MEMORY)

def build_prompt(history: List[HistoryMessage]) -> str:
    # Phi-2 responds better to an instruction/response format.
    out: List[str] = []
    if FACT_MEMORY:
        out.append("### Memory:\n")
        for fact in FACT_MEMORY[-10:]:
            out.append(f"- {fact}\n")
        out.append("\n")
    for m in history:
        if m["role"] == "user":
            out.append(
                "### Instruction:\n"
                f"{SYSTEM_PROMPT}\n\n"
                f"User: {m['content']}\n\n"
                "### Response:\n"
            )
        else:
            out.append(f"{m['content']}\n\n")
    return "".join(out)
