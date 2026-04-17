# SPDX-License-Identifier: Apache-2.0
"""S2-Pro tokenizer adapter wrapping HuggingFace PreTrainedTokenizerFast.

S2-Pro uses Qwen3 chat-format prompts built via the ``Conversation`` class:
- System message: reference text + VQ codes (voice cloning)
- User message: target text to synthesize
- Assistant message: ``<|voice|>`` modality marker (generation starts here)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.fishaudio_s2_pro.fish_speech.tokenizer import (
    IM_END_TOKEN,
    SEMANTIC_TOKEN_TEMPLATE,
)

logger = logging.getLogger(__name__)


@dataclass
class Reference:
    """A voice-cloning reference for S2-Pro TTS."""

    audio_bytes: bytes
    text: str
    vq_codes: torch.Tensor | None = None
    speaker: int | str | None = None


_SPEAKER_TAGGED_TURN_RE = re.compile(r"\[(S\d+)\]\s*", re.IGNORECASE)


def _canonical_speaker_id(value: int | str | None) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value

    text = str(value).strip()
    if not text:
        return None

    upper_text = text.upper()
    if upper_text.startswith("S") and upper_text[1:].isdigit():
        return int(upper_text[1:])
    if text.isdigit():
        return int(text)
    return text


def _parse_speaker_tagged_turns(text: str) -> list[tuple[int | str, str]]:
    cleaned = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_SPEAKER_TAGGED_TURN_RE.finditer(cleaned))
    if not matches:
        return []

    turns: list[tuple[int | str, str]] = []
    for idx, match in enumerate(matches):
        speaker_id = _canonical_speaker_id(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        turn_text = cleaned[start:end].strip()
        if speaker_id is not None and turn_text:
            turns.append((speaker_id, turn_text))
    return turns


class S2ProTokenizerAdapter:
    """
    Builds Qwen3 chat-format prompts using the ``Conversation`` class
    """

    def __init__(self, hf_tokenizer: PreTrainedTokenizerFast) -> None:
        self._tok = hf_tokenizer

    @property
    def eos_token_ids(self) -> list[int]:
        return [self._tok.convert_tokens_to_ids(IM_END_TOKEN)]

    @property
    def semantic_begin_id(self) -> int:
        return self._tok.convert_tokens_to_ids(SEMANTIC_TOKEN_TEMPLATE.format(i=0))

    @property
    def semantic_end_id(self) -> int:
        return self._tok.convert_tokens_to_ids(SEMANTIC_TOKEN_TEMPLATE.format(i=4095))

    def build_prompt(
        self,
        text: str,
        references: list[Reference] | None = None,
        *,
        num_codebooks: int = 10,
        speaker: int | str = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build an S2-Pro prompt using Qwen3 chat format."""
        from sglang_omni.models.fishaudio_s2_pro.fish_speech.content_sequence import (
            ContentSequence,
            TextPart,
            VQPart,
        )
        from sglang_omni.models.fishaudio_s2_pro.fish_speech.conversation import (
            Conversation,
            Message,
        )

        conversation = Conversation()

        # System message: reference audio for voice cloning
        if references:
            system_parts: list = []
            system_parts.append(
                TextPart(
                    text="convert the provided text to speech reference to the following:\n\nText:\n",
                    cal_loss=False,
                )
            )

            reference_seq = ContentSequence(modality="interleave")
            for ref in references:
                speaker_id = _canonical_speaker_id(ref.speaker)
                if speaker_id is None:
                    speaker_id = _canonical_speaker_id(speaker) or 0

                ref_parts: list[Any] = []
                if ref.text:
                    ref_parts.append(TextPart(text=ref.text, cal_loss=False))
                if ref.vq_codes is not None:
                    ref_parts.append(VQPart(codes=ref.vq_codes, cal_loss=False))
                if ref_parts:
                    reference_seq.append(ref_parts, speaker=speaker_id, add_end=True)

            system_parts.append(TextPart(text="\n\nSpeech:\n", cal_loss=False))
            system_parts.extend(reference_seq.parts)

            conversation.append(
                Message(
                    role="system",
                    parts=system_parts,
                    cal_loss=False,
                    add_im_start=True,
                    add_im_end=True,
                )
            )

        # User message: text to synthesize
        dialogue_turns = _parse_speaker_tagged_turns(text)
        if dialogue_turns:
            dialogue_seq = ContentSequence(modality="interleave")
            for speaker_id, turn_text in dialogue_turns:
                dialogue_seq.append(
                    TextPart(text=turn_text, cal_loss=False),
                    speaker=speaker_id,
                    add_end=True,
                )
            user_parts = dialogue_seq.parts
        else:
            text_with_tag = f"<|speaker:{speaker}|>{text}"
            user_parts = [TextPart(text=text_with_tag, cal_loss=False)]

        conversation.append(
            Message(
                role="user",
                parts=user_parts,
                cal_loss=False,
                add_im_start=True,
                add_im_end=True,
            )
        )

        # Assistant message: voice modality marker (generation starts after this)
        conversation.append(
            Message(
                role="assistant",
                parts=[],
                cal_loss=False,
                modality="voice",
                add_im_start=True,
                add_im_end=False,
            )
        )

        encoded = conversation.encode(self._tok, add_shift=False)
        vq_parts_list = encoded.vq_parts  # list of [num_codebooks, T_i]

        return {
            "input_ids": encoded.tokens,
            "vq_mask_tokens": encoded.vq_mask_tokens,
            "vq_parts": vq_parts_list,
        }
