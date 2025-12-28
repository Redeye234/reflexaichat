# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
# pyright: reportIndexIssue=false
# pyright: reportMissingTypeArgument=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
# pyright: reportDeprecated=false
from typing import TypedDict

import reflex as rx

from .state import ChatState

class Message(TypedDict):
    role: str
    content: str

def message_bubble(msg: rx.Var[Message]) -> rx.Component:
    is_user = msg["role"] == "user"  # type: ignore[index]
    is_note = msg["role"] == "note"  # type: ignore[index]
    return rx.box(
        rx.cond(
            is_note,
            rx.text(
                msg["content"],  # type: ignore[index]
                font_size="12px",
                color="#8a8175",
                text_align="center",
                width="100%",
            ),
            rx.cond(
                is_user,
                rx.text(
                    msg["content"],  # type: ignore[index]
                    font_size="15px",
                    line_height="1.6",
                    color="#1f1c17",
                    white_space="pre-wrap",
                ),
                rx.markdown(
                    msg["content"],  # type: ignore[index]
                    font_size="15px",
                    line_height="1.6",
                    color="#1f1c17",
                ),
            ),
        ),
        bg=rx.cond(
            is_note,
            "transparent",
            rx.cond(is_user, "rgba(255,255,255,0.85)", "rgba(255,255,255,0.98)"),
        ),
        border=rx.cond(is_note, "none", "1px solid rgba(0,0,0,0.08)"),
        border_radius=rx.cond(is_note, "0px", "16px"),
        padding=rx.cond(is_note, "6px 0", "12px 14px"),
        max_width=rx.cond(is_note, "100%", "85%"),
        overflow_x="auto",
        align_self=rx.cond(
            is_note,
            "center",
            rx.cond(is_user, "flex-end", "flex-start"),
        ),
        box_shadow=rx.cond(is_note, "none", "0 8px 18px rgba(0,0,0,0.08)"),
    )

def index() -> rx.Component:
    return rx.box(
        rx.el.link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap",
        ),
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.vstack(
                        rx.text(
                            "Reflex AI Chat",
                            font_size=["28px", "32px", "36px"],
                            font_weight="600",
                            letter_spacing="-0.02em",
                        ),
                        rx.text(
                            "Local GGUF chat with lightweight memory",
                            color="var(--muted)",
                            font_size="13px",
                        ),
                        spacing="1",
                    ),
                    rx.hstack(
                        rx.box(
                            width="8px",
                            height="8px",
                            border_radius="999px",
                            bg="var(--accent)",
                        ),
                        rx.text("Local", font_size="12px", color="var(--muted)"),
                        bg="rgba(255,255,255,0.7)",
                        border="1px solid rgba(0,0,0,0.08)",
                        padding="6px 12px",
                        border_radius="999px",
                        spacing="2",
                        align_items="center",
                    ),
                    justify="between",
                    align="center",
                    width="100%",
                ),
                rx.box(
                    rx.vstack(
                        rx.foreach(ChatState.messages, message_bubble),
                        rx.cond(
                            ChatState.is_streaming,
                            rx.hstack(
                                rx.box(
                                    width="8px",
                                    height="8px",
                                    border_radius="999px",
                                    bg="var(--ink)",
                                    animation="pulse 1.2s ease-in-out infinite",
                                ),
                                rx.text(
                                    "Typing...",
                                    font_size="13px",
                                    color="var(--muted)",
                                ),
                                spacing="2",
                                align_items="center",
                            ),
                        ),
                        spacing="3",
                        width="100%",
                    ),
                    bg="rgba(255,255,255,0.78)",
                    border="1px solid rgba(0,0,0,0.08)",
                    border_radius="24px",
                    padding="24px",
                    width="100%",
                    height=["56vh", "60vh", "64vh"],
                    overflow_y="auto",
                    box_shadow="0 30px 80px rgba(0,0,0,0.12)",
                    style={"backdropFilter": "blur(20px)"},
                ),
                rx.hstack(
                    rx.text_area(
                        placeholder="Ask something...",
                        value=ChatState.input_text,
                        on_change=ChatState.set_input_text,  # type: ignore[arg-type, attr-defined]
                        on_key_down=ChatState.handle_key,  # type: ignore[arg-type]
                        width="100%",
                        debounce_timeout=0,
                        min_height="84px",
                        bg="rgba(255,255,255,0.95)",
                        border="1px solid rgba(0,0,0,0.12)",
                        border_radius="20px",
                        padding="16px 18px",
                        font_size="15px",
                        box_shadow="inset 0 1px 2px rgba(0,0,0,0.06)",
                    ),
                    rx.vstack(
                        rx.button(
                            "Send",
                            on_click=ChatState.send,  # type: ignore[arg-type]
                            bg="var(--accent)",
                            color="#ffffff",
                            border_radius="14px",
                            padding="12px 18px",
                            width="100%",
                        ),
                        rx.button(
                            "Stop",
                            on_click=ChatState.stop_generation,  # type: ignore[arg-type]
                            variant="outline",
                            border_radius="14px",
                            color="#a1463f",
                            border_color="rgba(161,70,63,0.5)",
                            disabled=rx.cond(ChatState.is_streaming, False, True),
                            width="100%",
                        ),
                        rx.button(
                            "Reset",
                            on_click=ChatState.reset_chat,  # type: ignore[arg-type]
                            variant="outline",
                            border_radius="14px",
                            disabled=ChatState.is_streaming,
                            width="100%",
                        ),
                        spacing="2",
                        width=["100%", "160px", "180px"],
                    ),
                    width="100%",
                    spacing="3",
                    align_items="stretch",
                ),
                spacing="4",
                width="100%",
            ),
            width="min(1100px, 92vw)",
            margin="0 auto",
            padding_top="24px",
            padding_bottom="40px",
        ),
        bg="radial-gradient(circle at top left, #f7f7f9 0%, #f2eee7 35%, #f7f6f3 70%, #ece8e1 100%)",
        min_height="100vh",
        font_family="Manrope, -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'SF Pro Display', system-ui, sans-serif",
        style={
            "--ink": "#1f1c17",
            "--muted": "#6a6256",
            "--accent": "#0a84ff",
            "@keyframes pulse": {
                "0%": {"transform": "scale(0.9)", "opacity": "0.4"},
                "50%": {"transform": "scale(1.0)", "opacity": "1"},
                "100%": {"transform": "scale(0.9)", "opacity": "0.4"},
            }
        },
    )

app = rx.App()
app.add_page(index, route="/")
