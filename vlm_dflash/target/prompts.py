"""Prompt rendering shared by regen, capture and eval (single code path)."""

from __future__ import annotations


def strip_image_placeholder(question: str) -> str:
    return question.replace("<image>", "").strip()


def render_prompt(processor, question: str) -> str:
    """Render the Qwen-VL chat prompt for a single image + text question."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": strip_image_placeholder(question)},
            ],
        }
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_prompt(processor, question: str, image) -> dict:
    """Tokenize + preprocess one image-text prompt; returns processor tensors."""
    text = render_prompt(processor, question)
    return processor(text=[text], images=[image], return_tensors="pt")
