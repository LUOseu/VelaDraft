"""Turn-start fallback for multi-turn capture spans (Qwen3.5/3.6 templates)."""

from types import SimpleNamespace

import pytest

from tests.test_wp2_multiturn import _FakeChatProcessor, _stored_last_turn_row
from vlm_dflash.target.sample_capture import _prepare_multi_turn, _turn_start_positions

_TURN_START = 4  # stands in for <|im_start|>
_THINK = 5  # the empty thinking block only the trailing assistant turn gets


class _TrailingThinkProcessor(_FakeChatProcessor):
    """Fake Qwen3.6-style template: every turn opens with a turn-start marker,
    and a conversation's trailing assistant turn (or the generation prompt)
    carries an empty thinking block that historical assistant turns lack."""

    unk_token_id = 0

    def __init__(
        self, marker: str | None = "<|im_start|>", trailing_user_drift: bool = False, unmarked_turn: int | None = None
    ):
        super().__init__()
        self.marker = marker
        self.trailing_user_drift = trailing_user_drift
        self.unmarked_turn = unmarked_turn

    def convert_tokens_to_ids(self, token):
        return _TURN_START if token == self.marker else self.unk_token_id

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        assert not tokenize
        parts = []
        for index, message in enumerate(messages):
            assistant = message["role"] != "user"
            parts.append(("" if index == self.unmarked_turn else chr(_TURN_START)) + ("\x02" if assistant else "\x01"))
            if assistant and index == len(messages) - 1 and not add_generation_prompt:
                parts.append(chr(_THINK))
            content = message["content"]
            if isinstance(content, list):
                parts.extend("<IMG>" if part["type"] == "image" else part["text"] for part in content)
            else:
                parts.append(content)
            parts.append("\x03")
        if self.trailing_user_drift and messages[-1]["role"] == "user" and not add_generation_prompt:
            parts.append("\x06")  # a template bug: prefixes ending in a user turn drift too
        if add_generation_prompt:
            parts.append(chr(_TURN_START) + "\x02" + chr(_THINK))
        return "".join(parts)


def test_turn_start_positions_follow_im_start_markers():
    ids = [4, 1, 2, 4, 3, 4, 6, 7, 4, 8]
    assert _turn_start_positions(_TrailingThinkProcessor(), ids, 4) == [0, 3, 5, 8]
    assert _turn_start_positions(_TrailingThinkProcessor(), ids, 3) is None  # marker count must match the turns
    assert _turn_start_positions(_TrailingThinkProcessor(marker=None), ids, 4) is None
    assert _turn_start_positions(SimpleNamespace(tokenizer=None), ids, 4) is None


def test_historical_assistant_turns_end_at_the_next_turn_start(tmp_path):
    processor = _TrailingThinkProcessor()
    row = _stored_last_turn_row(tmp_path, processor, "two")
    prepared = _prepare_multi_turn(processor, row, tmp_path, seq_length=4096)
    full_ids = row["regen"]["full_ids"]
    starts = [index for index, token in enumerate(full_ids) if token == _TURN_START]
    assert len(starts) == 4
    # A prefix ending at assistant turn 1 would render it as trailing (with the
    # thinking block), so its boundary comes from the next turn-start marker;
    # the prefixes ending in user turns are still render-checked.
    bounds = [*starts, len(full_ids)]
    assert prepared["meta"]["turn_spans"] == [[bounds[i], bounds[i + 1]] for i in range(4)]


def test_prefix_drift_without_turn_start_markers_still_fails(tmp_path):
    processor = _TrailingThinkProcessor(marker=None)
    row = _stored_last_turn_row(tmp_path, processor, "two")
    with pytest.raises(RuntimeError, match="Prefix render drift .* at turn 1"):
        _prepare_multi_turn(processor, row, tmp_path, seq_length=4096)


def test_user_turn_prefix_drift_still_fails_with_markers(tmp_path):
    processor = _TrailingThinkProcessor(trailing_user_drift=True)
    row = _stored_last_turn_row(tmp_path, processor, "two")
    with pytest.raises(RuntimeError, match="Prefix render drift .* at turn 0"):
        _prepare_multi_turn(processor, row, tmp_path, seq_length=4096)


def test_markers_must_agree_with_user_turn_prefix_renders(tmp_path, monkeypatch):
    import tests.test_wp2_multiturn as wp2

    # A literal marker in the dataset text plus a turn the template renders
    # without one keeps the count at one per turn but misplaces the boundaries.
    base_row = wp2._multiturn_row

    def row_with_literal_marker():
        row = base_row()
        row["conversations"][0]["value"] += chr(_TURN_START)
        return row

    monkeypatch.setattr(wp2, "_multiturn_row", row_with_literal_marker)
    processor = _TrailingThinkProcessor(unmarked_turn=2)
    row = _stored_last_turn_row(tmp_path, processor, "two")
    assert row["regen"]["full_ids"].count(_TURN_START) == 4
    with pytest.raises(RuntimeError, match="disagree with the prefix render at turn 0"):
        _prepare_multi_turn(processor, row, tmp_path, seq_length=4096)
