#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for ElevenLabs TTS alignment and session handling."""

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from websockets.protocol import State

from pipecat.services.elevenlabs.tts import (
    ElevenLabsTTSService,
    _select_alignment,
    _strip_utterance_leading_spaces,
    calculate_word_times,
)

_WS_ALIGNMENT_KEYS = ("chars", "charStartTimesMs", "charDurationsMs")


def _chunk(text: str) -> dict[str, list[Any]]:
    chars = list(text)
    return {
        "chars": chars,
        "charStartTimesMs": [i * 100 for i in range(len(chars))],
        "charDurationsMs": [100 for _ in chars],
    }


def _words_from_chunks(chunks: list[dict[str, list[Any]]]) -> list[str]:
    cumulative_time = 0.0
    partial_word = ""
    partial_word_start_time = 0.0
    word_times = []
    alignment_started = False

    for chunk in chunks:
        alignment = _strip_utterance_leading_spaces(
            chunk,
            _WS_ALIGNMENT_KEYS,
            not alignment_started,
        )
        alignment_started = True
        chunk_word_times, partial_word, partial_word_start_time = calculate_word_times(
            alignment,
            cumulative_time,
            partial_word,
            partial_word_start_time,
        )
        word_times.extend(chunk_word_times)

        starts = alignment["charStartTimesMs"]
        durations = alignment["charDurationsMs"]
        if starts and durations:
            cumulative_time += (starts[-1] + durations[-1]) / 1000.0

    if partial_word:
        word_times.append((partial_word, partial_word_start_time))

    return [word for word, _ in word_times]


class _FakeWebSocket:
    def __init__(self):
        self.state = State.OPEN
        self.sent_messages: list[dict[str, Any]] = []

    async def send(self, message: str):
        self.sent_messages.append(json.loads(message))

    async def close(self):
        self.state = State.CLOSED


async def _noop(*args, **kwargs):
    return None


def test_elevenlabs_flash_alignment_preserves_inter_word_chunk_space():
    chunks = [
        _chunk(" Why did the math book"),
        _chunk(" look so sad? "),
        _chunk(" Because it had too m"),
        _chunk("any problems. "),
    ]

    assert _words_from_chunks(chunks) == [
        "Why",
        "did",
        "the",
        "math",
        "book",
        "look",
        "so",
        "sad?",
        "Because",
        "it",
        "had",
        "too",
        "many",
        "problems.",
    ]


def test_elevenlabs_alignment_strips_only_utterance_leading_spaces():
    first = _strip_utterance_leading_spaces(_chunk("  Hello"), _WS_ALIGNMENT_KEYS, True)
    subsequent = _strip_utterance_leading_spaces(_chunk(" world"), _WS_ALIGNMENT_KEYS, False)

    assert first["chars"] == list("Hello")
    assert subsequent["chars"] == list(" world")


def test_select_alignment_default_prefers_alignment():
    msg = {
        "alignment": _chunk("Hello"),
        "normalizedAlignment": _chunk(" Hello"),
    }
    selected = _select_alignment(
        msg,
        normalized_key="normalizedAlignment",
        alignment_key="alignment",
        prefer_normalized=False,
    )
    assert selected is not None
    assert selected["chars"] == list("Hello")


def test_select_alignment_dictionary_mode_prefers_normalized():
    msg = {
        "alignment": _chunk("Hello"),
        "normalizedAlignment": _chunk(" Hello"),
    }
    selected = _select_alignment(
        msg,
        normalized_key="normalizedAlignment",
        alignment_key="alignment",
        prefer_normalized=True,
    )
    assert selected is not None
    assert selected["chars"] == list(" Hello")


def test_select_alignment_falls_back_when_preferred_missing():
    msg_default = {"normalizedAlignment": _chunk(" Hello")}
    selected = _select_alignment(
        msg_default,
        normalized_key="normalizedAlignment",
        alignment_key="alignment",
        prefer_normalized=False,
    )
    assert selected is not None
    assert selected["chars"] == list(" Hello")

    msg_dict = {"alignment": _chunk("Hello")}
    selected = _select_alignment(
        msg_dict,
        normalized_key="normalizedAlignment",
        alignment_key="alignment",
        prefer_normalized=True,
    )
    assert selected is not None
    assert selected["chars"] == list("Hello")


def test_select_alignment_falls_back_when_preferred_null():
    msg = {"alignment": None, "normalizedAlignment": _chunk(" Hello")}
    selected = _select_alignment(
        msg,
        normalized_key="normalizedAlignment",
        alignment_key="alignment",
        prefer_normalized=False,
    )
    assert selected is not None
    assert selected["chars"] == list(" Hello")


def test_select_alignment_returns_none_when_both_missing():
    assert (
        _select_alignment(
            {},
            normalized_key="normalizedAlignment",
            alignment_key="alignment",
            prefer_normalized=False,
        )
        is None
    )
    assert (
        _select_alignment(
            {"alignment": None, "normalizedAlignment": None},
            normalized_key="normalizedAlignment",
            alignment_key="alignment",
            prefer_normalized=True,
        )
        is None
    )


def test_select_alignment_works_with_http_field_names():
    msg = {
        "alignment": {"characters": list("Hi")},
        "normalized_alignment": {"characters": list(" Hi")},
    }
    selected = _select_alignment(
        msg,
        normalized_key="normalized_alignment",
        alignment_key="alignment",
        prefer_normalized=False,
    )
    assert selected is not None
    assert selected["characters"] == list("Hi")

    selected = _select_alignment(
        msg,
        normalized_key="normalized_alignment",
        alignment_key="alignment",
        prefer_normalized=True,
    )
    assert selected is not None
    assert selected["characters"] == list(" Hi")


@pytest.mark.asyncio
async def test_voice_settings_are_sent_only_once_per_websocket_session():
    service = ElevenLabsTTSService(
        api_key="test-key",
        settings=ElevenLabsTTSService.Settings(
            voice="voice-id",
            model="eleven_turbo_v2_5",
            stability=0.7,
            similarity_boost=0.8,
            speed=1.1,
        ),
    )
    service._websocket = _FakeWebSocket()
    service.start_ttfb_metrics = _noop
    service.start_tts_usage_metrics = _noop

    async for _ in service.run_tts("hello there", "ctx-1"):
        pass
    async for _ in service.run_tts("hello again", "ctx-2"):
        pass

    init_messages = [msg for msg in service._websocket.sent_messages if msg["text"] == " "]
    assert len(init_messages) == 2
    assert init_messages[0]["voice_settings"] == {
        "stability": 0.7,
        "similarity_boost": 0.8,
        "speed": 1.1,
    }
    assert "voice_settings" not in init_messages[1]


@pytest.mark.asyncio
async def test_disconnect_resets_voice_settings_session_marker():
    service = ElevenLabsTTSService(
        api_key="test-key",
        settings=ElevenLabsTTSService.Settings(
            voice="voice-id",
            model="eleven_turbo_v2_5",
            stability=0.7,
            similarity_boost=0.8,
            speed=1.1,
        ),
    )
    service._websocket = _FakeWebSocket()
    service._voice_settings_sent = True
    service.stop_all_metrics = AsyncMock()
    service.remove_active_audio_context = AsyncMock()
    service._call_event_handler = AsyncMock()

    await service._disconnect_websocket()

    assert service._voice_settings_sent is False
    assert service._websocket is None


@pytest.mark.asyncio
async def test_voice_settings_update_forces_reconnect():
    service = ElevenLabsTTSService(
        api_key="test-key",
        settings=ElevenLabsTTSService.Settings(
            voice="voice-id",
            model="eleven_turbo_v2_5",
            stability=0.7,
            similarity_boost=0.8,
            speed=1.1,
        ),
    )
    service._disconnect = AsyncMock()
    service._connect = AsyncMock()

    await service._update_settings(ElevenLabsTTSService.Settings(speed=0.9))

    service._disconnect.assert_awaited_once()
    service._connect.assert_awaited_once()
