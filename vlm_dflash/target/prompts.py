"""Prompt rendering shared by regen, capture and eval (single code path).

Two shapes exist: :func:`render_prompt` / :func:`encode_prompt` for v1 rows
(single image, single question - byte-identical to the original pipeline) and
:func:`messages_from_row` / :func:`render_conversation` / :func:`encode_conversation`
for schema-v2 rows (multi-image, multi-turn). The v2 renderer places each
turn's content from the row's ordered ``segments`` when present (text and
images keep their declared positions); legacy rows without segments keep the
historical layout (images at the START of the turn, ``<image>`` placeholders
stripped), which is exactly the v1 layout for single-image rows.
"""

from __future__ import annotations


_HUMAN_ROLES = ("human", "user")
_GPT_ROLES = ("gpt", "assistant", "model")

#: Processor attribute with the contract's chat-template kwargs (set by
#: ``VLMAdapter.finalize_processor``; see docs/REFERENCE.md).
CHAT_TEMPLATE_KWARGS_ATTR = "vlm_dflash_chat_template_kwargs"


def _apply_chat_template(processor, messages: list[dict], *, add_generation_prompt: bool) -> str:
    kwargs = getattr(processor, CHAT_TEMPLATE_KWARGS_ATTR, {})
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs
    )


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
    return _apply_chat_template(processor, messages, add_generation_prompt=True)


def encode_prompt(processor, question: str, image) -> dict:
    """Tokenize + preprocess one image-text prompt; returns processor tensors."""
    text = render_prompt(processor, question)
    return processor(text=[text], images=[image], return_tensors="pt")


def messages_from_row(
    row: dict,
    *,
    assistant_overrides: dict[int, str] | None = None,
    upto: int | None = None,
) -> tuple[list[dict], list[int]]:
    """Build HF chat messages for a schema-v2 row.

    Returns ``(messages, image_indices)`` where ``image_indices`` are positions
    into ``row["images"]`` for every image attached to a turn included in
    ``messages`` (appearance order). ``assistant_overrides`` replaces the text
    of specific assistant turns by conversation index (regen uses the generated
    text as teacher-forced history); ``upto`` truncates the conversation.
    """
    conversations = row.get("conversations") or []
    if not conversations:
        raise ValueError(f"Row {row.get('id')!r} has no conversations.")
    images = row.get("images") or []
    if isinstance(images, str):
        images = [images]
    image_turns = row.get("image_turns")
    if image_turns is None:
        image_turns = [0] * len(images)
    if len(image_turns) != len(images):
        raise ValueError(
            f"Row {row.get('id')!r} has {len(images)} image(s) but {len(image_turns)} image_turns entr(ies)."
        )
    per_turn: dict[int, list[int]] = {}
    for position, turn in enumerate(image_turns):
        per_turn.setdefault(int(turn), []).append(position)
    limit = len(conversations) if upto is None else min(int(upto), len(conversations))
    messages: list[dict] = []
    image_indices: list[int] = []
    for index, turn in enumerate(conversations[:limit]):
        role = str(turn.get("from", "")).lower()
        text = str(turn.get("value", ""))
        if role in _HUMAN_ROLES:
            parts: list[dict] = []
            segments = turn.get("segments")
            if segments is not None:
                attached = per_turn.get(index, [])
                for segment in segments:
                    if segment.get("type") == "image":
                        position = int(segment["index"])
                        if not 0 <= position < len(images):
                            raise ValueError(
                                f"Row {row.get('id')!r} references image {position}, but only "
                                f"{len(images)} image(s) exist."
                            )
                        if position not in attached:
                            raise ValueError(
                                f"Row {row.get('id')!r} places image {position} on turn {index}, but its "
                                "image_turns entry attaches it to another turn."
                            )
                        parts.append({"type": "image"})
                        image_indices.append(position)
                    else:
                        segment_text = str(segment.get("text", ""))
                        if segment_text:
                            parts.append({"type": "text", "text": segment_text})
            else:
                # Legacy rows (pre-segment): images at the turn start, placeholders stripped.
                for position in per_turn.get(index, []):
                    parts.append({"type": "image"})
                    image_indices.append(position)
                stripped = strip_image_placeholder(text)
                if stripped or not parts:
                    parts.append({"type": "text", "text": stripped})
            if not parts:
                parts.append({"type": "text", "text": ""})
            messages.append({"role": "user", "content": parts})
        elif role in _GPT_ROLES:
            if assistant_overrides is not None and index in assistant_overrides:
                text = assistant_overrides[index]
            messages.append({"role": "assistant", "content": text})
        else:
            raise ValueError(f"Row {row.get('id')!r} has unsupported role {role!r} at turn {index}.")
    return messages, image_indices


def same_text_render(tokenizer, committed_ids: list[int], rendered_ids: list[int], *, protected_ids=()) -> bool:
    """Whether ``committed_ids`` is ``rendered_ids`` up to a tokenizer non-roundtrip split.

    A tokenizer does not guarantee encode/decode roundtrip token identity: a
    sampled answer can be a non-canonical split of its own text, so a fresh
    render of the same conversation tokenizes it differently. That is the same
    text, not prompt drift. Accepted when both sequences decode (special tokens
    kept) to the same text, up to trailing whitespace the render adds after the
    committed ids, and no ``protected_ids`` (image tokens, whose positions the
    multimodal inputs encode) occur from the first differing token on.
    """
    committed, rendered = [int(token) for token in committed_ids], [int(token) for token in rendered_ids]
    first = next(
        (index for index, (left, right) in enumerate(zip(committed, rendered)) if left != right),
        min(len(committed), len(rendered)),
    )
    protected = {int(token) for token in protected_ids}
    if protected & (set(committed[first:]) | set(rendered[first:])):
        return False
    committed_text = tokenizer.decode(committed, skip_special_tokens=False)
    rendered_text = tokenizer.decode(rendered, skip_special_tokens=False)
    return rendered_text.startswith(committed_text) and not rendered_text[len(committed_text) :].strip()


def multimodal_token_ids(processor, inputs) -> set[int]:
    """Image/video token ids of a render: the processor's own ids plus every ``mm_token_type_ids != 0`` token."""
    ids = {
        int(value)
        for value in (getattr(processor, "image_token_id", None), getattr(processor, "video_token_id", None))
        if isinstance(value, int)
    }
    mm_types = inputs.get("mm_token_type_ids")
    if mm_types is not None:
        ids.update(int(token) for token in inputs["input_ids"][mm_types != 0].tolist())
    return ids


def render_conversation(processor, messages: list[dict], *, add_generation_prompt: bool = False) -> str:
    """Chat-template text for a (possibly truncated) conversation."""
    return _apply_chat_template(processor, messages, add_generation_prompt=add_generation_prompt)


def encode_conversation(processor, messages: list[dict], images: list, *, add_generation_prompt: bool = False) -> dict:
    """Tokenize + preprocess a conversation; ``images`` in the messages' appearance order."""
    text = render_conversation(processor, messages, add_generation_prompt=add_generation_prompt)
    return processor(text=[text], images=list(images), return_tensors="pt")
