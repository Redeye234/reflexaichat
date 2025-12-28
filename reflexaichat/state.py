from typing import List, Dict, Any, Iterable, Iterator, cast, TypedDict

import reflex as rx

from llm_core import HistoryMessage, build_prompt, maybe_store_fact, get_llm, reset_memory

class QueuedMessage(TypedDict):
    text: str
    already_added: bool

class ChatState(rx.State):
    messages: List[Dict[str, str]] = []
    input_text: str = ""
    is_streaming: bool = False
    stop_requested: bool = False
    pending_messages: List[QueuedMessage] = []
    ignore_next_change: bool = False

    def reset_chat(self) -> None:
        self.messages = []
        self.input_text = ""
        reset_memory()
        self.is_streaming = False
        self.stop_requested = False
        self.pending_messages = []
        self.ignore_next_change = False

    def stop_generation(self) -> None:
        self.stop_requested = True

    def handle_key(self, key: str) -> Iterator[Any]:
        if key == "Enter":
            self.ignore_next_change = True
            yield from self.send()

    def set_input_text(self, value: str) -> None:
        if self.ignore_next_change:
            self.ignore_next_change = False
            return
        self.input_text = value

    def _prompt_history(self) -> List[HistoryMessage]:
        history = [m for m in self.messages if m["role"] in ("user", "assistant")]
        return cast(List[HistoryMessage], history)

    def _run_message(self, text: str, already_added: bool) -> Iterator[Any]:
        if not already_added:
            self.messages = self.messages + [{"role": "user", "content": text}]
            maybe_store_fact(text)

        # Add a placeholder assistant message for streaming updates.
        self.messages = self.messages + [{"role": "assistant", "content": ""}]
        self.is_streaming = True
        self.stop_requested = False
        yield

        prompt_history = self._prompt_history()[:-1]
        prompt = build_prompt(prompt_history)
        llm = get_llm()
        stream = llm(
            prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repeat_penalty=1.05,
            stop=["### Instruction:", "User:", "Assistant:"],
            stream=True,
        )

        chunks: List[str] = []
        for chunk in cast(Iterable[Dict[str, Any]], stream):
            if self.stop_requested:
                break
            text_chunk = cast(str, chunk["choices"][0]["text"])
            if not text_chunk:
                continue
            chunks.append(text_chunk)
            updated = list(self.messages)
            updated[-1] = {"role": "assistant", "content": "".join(chunks)}
            self.messages = updated
            yield

        if self.stop_requested:
            self.messages = self.messages + [{"role": "note", "content": "Stopped."}]

        self.is_streaming = False
        self.stop_requested = False
        yield

    def send(self) -> Iterator[Any]:
        text = self.input_text.strip()
        if not text:
            return

        self.input_text = ""
        if self.is_streaming:
            self.messages = self.messages + [{"role": "user", "content": text}]
            maybe_store_fact(text)
            self.pending_messages = self.pending_messages + [
                {"text": text, "already_added": True}
            ]
            return

        yield from self._run_message(text, already_added=False)
        while self.pending_messages:
            next_msg = self.pending_messages[0]
            self.pending_messages = self.pending_messages[1:]
            yield from self._run_message(
                next_msg["text"], already_added=next_msg["already_added"]
            )
