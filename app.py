import sys
import time
from typing import List, Dict, Iterable, cast, Any

from llm_core import (
    MODEL_PATH,
    N_CTX,
    N_GPU_LAYERS,
    N_THREADS,
    HistoryMessage,
    build_prompt,
    maybe_store_fact,
    get_llm,
    get_memory,
    reset_memory,
)

def main():
    print("Loading model...")
    print(f"  path={MODEL_PATH}")
    print(f"  n_ctx={N_CTX} | n_gpu_layers={N_GPU_LAYERS} | n_threads={N_THREADS}")

    try:
        llm = get_llm()
    except FileNotFoundError as exc:
        print(str(exc))
        print("Put your GGUF file at models/model.gguf or set MODEL_PATH env var.")
        sys.exit(1)

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
            reset_memory()
            print("(history cleared)\n")
            continue
        if user_text.lower() in ("/memory", "memory"):
            memory = get_memory()
            if memory:
                print("\nMemory:")
                for fact in memory[-10:]:
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
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repeat_penalty=1.05,
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
