import os
import sys
import time
from pathlib import Path
import re
from typing import List, Dict, Iterable, cast, TypedDict, Any

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

def maybe_store_fact(text: str) -> None:
    match = FACT_PATTERN.match(text)
    if not match:
        return
    subject = match.group(1).strip()
    predicate = match.group(3).strip().rstrip(".")
    fact = f"{subject} is {predicate}"
    if fact not in FACT_MEMORY:
        FACT_MEMORY.append(fact)

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

def main():
    path = Path(MODEL_PATH)
    if not path.exists():
        print(f"Model not found: {path.resolve()}")
        print("Put your GGUF file at models/model.gguf or set MODEL_PATH env var.")
        sys.exit(1)

    print("Loading model...")
    print(f"  path={path.resolve()}")
    print(f"  n_ctx={N_CTX} | n_gpu_layers={N_GPU_LAYERS} | n_threads={N_THREADS}")

    llm = Llama(
        model_path=str(path),
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,   # 0 => CPU-only
        n_threads=N_THREADS,
        verbose=False,
    )

    history: List[HistoryMessage] = []

    print("\nLocal LLM chat started. Type /exit to quit, /reset to clear history.\n")

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in ("/exit", "exit", "quit"):
            break
        if user_text.lower() in ("/reset", "reset"):
            history.clear()
            FACT_MEMORY.clear()
            print("(history cleared)\n")
            continue
        if user_text.lower() in ("/memory", "memory"):
            if FACT_MEMORY:
                print("\nMemory:")
                for fact in FACT_MEMORY[-10:]:
                    print(f"- {fact}")
                print("")
            else:
                print("\nMemory is empty.\n")
            continue

        history.append(cast(HistoryMessage, {"role": "user", "content": user_text}))
        maybe_store_fact(user_text)
        prompt = build_prompt(history)

        t0 = time.time()
        # Generate (streaming)
        stream = llm(
            prompt,
            max_tokens=256,
            temperature=0.2,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            stop=["### Instruction:", "User:", "Assistant:"],  # stop before a new turn
            stream=True,
        )

        print("\nAssistant: ", end="", flush=True)
        chunks: List[str] = []
        prefix_stripped = False
        for chunk in cast(Iterable[Dict[str, Any]], stream):
            text = cast(str, chunk["choices"][0]["text"])
            if text:
                if not prefix_stripped:
                    text = text.lstrip()
                    if text.startswith("Assistant:"):
                        text = text[len("Assistant:"):].lstrip()
                    prefix_stripped = True
                chunks.append(text)
                print(text, end="", flush=True)

        dt = time.time() - t0
        assistant_text = "".join(chunks).strip()
        history.append(cast(HistoryMessage, {"role": "assistant", "content": assistant_text}))

        print(f"\n\n(Completed in {dt:.2f}s)\n")

if __name__ == "__main__":
    main()
