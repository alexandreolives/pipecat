#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from unittest.mock import AsyncMock

import pytest

from pipecat.frames.frames import (
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    UserTurnInferenceCompletedFrame,
    UserTurnInferenceTriggeredFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.services.deepgram.flux.base import (
    DeepgramFluxSTTBase,
    DeepgramFluxSTTSettings,
)
from pipecat.turns.user_stop import ExternalUserTurnCompletionStopStrategy


class DummyDeepgramFluxSTTService(DeepgramFluxSTTBase):
    async def run_stt(self, audio: bytes):
        yield None

    async def _transport_send_audio(self, audio: bytes):
        return None

    async def _transport_send_json(self, message: dict):
        return None

    def _transport_is_active(self) -> bool:
        return True

    async def _connect(self):
        return None

    async def _disconnect(self):
        return None


def _make_service() -> DummyDeepgramFluxSTTService:
    service = DummyDeepgramFluxSTTService(
        encoding="linear16",
        sample_rate=16000,
        settings=DeepgramFluxSTTSettings(model="flux-general-en", min_confidence=None),
    )
    service._user_id = "user-1"
    service.push_frame = AsyncMock()
    service.broadcast_frame = AsyncMock()
    service.broadcast_interruption = AsyncMock()
    service.stop_processing_metrics = AsyncMock()
    service._call_event_handler = AsyncMock()
    service._handle_transcription = AsyncMock()
    return service


@pytest.mark.asyncio
async def test_eager_end_of_turn_emits_provisional_transcript_and_inference_trigger():
    service = _make_service()

    await service._handle_eager_end_of_turn("hello there", {})

    assert service.push_frame.await_count == 2
    first_call = service.push_frame.await_args_list[0].args[0]
    second_call = service.push_frame.await_args_list[1].args[0]

    assert isinstance(first_call, TranscriptionFrame)
    assert first_call.text == "hello there"
    assert first_call.finalized is False
    assert isinstance(second_call, UserTurnInferenceTriggeredFrame)

    service._call_event_handler.assert_awaited_once_with("on_eager_end_of_turn", "hello there")


@pytest.mark.asyncio
async def test_turn_resumed_interrupts_inflight_eager_response_once():
    service = _make_service()
    service._eager_eot_transcript = "hello there"
    service._eager_eot_interruption_sent = False

    await service._handle_turn_resumed("TurnResumed")
    await service._handle_turn_resumed("TurnResumed")

    assert service.broadcast_interruption.await_count == 1
    assert service._eager_eot_interruption_sent is True
    assert service._call_event_handler.await_count == 2


@pytest.mark.asyncio
async def test_turn_resumed_interrupts_even_when_start_interrupts_are_external():
    service = _make_service()
    service._should_interrupt = False
    service._eager_eot_transcript = "hello there"
    service._eager_eot_interruption_sent = False

    await service._handle_turn_resumed("TurnResumed")

    service.broadcast_interruption.assert_awaited_once()
    assert service._eager_eot_interruption_sent is True


@pytest.mark.asyncio
async def test_end_of_turn_emits_full_transcript_and_completion_frames():
    service = _make_service()
    service._eager_eot_transcript = "hello"

    await service._handle_end_of_turn("hello world", {})

    assert service.push_frame.await_count == 2
    first_call = service.push_frame.await_args_list[0].args[0]
    second_call = service.push_frame.await_args_list[1].args[0]

    assert isinstance(first_call, TranscriptionFrame)
    assert first_call.text == "hello world"
    assert first_call.finalized is True
    assert isinstance(second_call, UserTurnInferenceCompletedFrame)
    service.broadcast_frame.assert_awaited_once_with(UserStoppedSpeakingFrame)


@pytest.mark.asyncio
async def test_eager_end_of_turn_respects_min_confidence_gate():
    service = _make_service()
    service._settings.min_confidence = 0.75

    await service._handle_eager_end_of_turn(
        "hello there",
        {
            "words": [
                {"confidence": 0.2},
                {"confidence": 0.3},
            ]
        },
    )

    assert service.push_frame.await_count == 0
    assert all(
        not isinstance(call.args[0], UserTurnInferenceTriggeredFrame)
        for call in service.push_frame.await_args_list
    )


@pytest.mark.asyncio
async def test_end_of_turn_respects_min_confidence_gate():
    service = _make_service()
    service._settings.min_confidence = 0.75

    await service._handle_end_of_turn(
        "hello there",
        {
            "words": [
                {"confidence": 0.2},
                {"confidence": 0.3},
            ]
        },
    )

    assert service.push_frame.await_count == 0
    service._handle_transcription.assert_not_awaited()
    service.broadcast_frame.assert_awaited_once_with(UserStoppedSpeakingFrame)


@pytest.mark.asyncio
async def test_external_completion_strategy_handles_eager_and_final_frames():
    strategy = ExternalUserTurnCompletionStopStrategy(enable_user_speaking_frames=False)

    inference_events: list[str] = []
    stopped_events: list[str] = []

    @strategy.event_handler("on_user_turn_inference_triggered")
    async def on_user_turn_inference_triggered(strategy):
        inference_events.append("triggered")

    @strategy.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(strategy, params):
        stopped_events.append("stopped")

    assert await strategy.process_frame(UserTurnInferenceTriggeredFrame()) == ProcessFrameResult.CONTINUE
    assert inference_events == ["triggered"]
    assert stopped_events == []

    assert await strategy.process_frame(UserTurnInferenceCompletedFrame()) == ProcessFrameResult.STOP
    assert inference_events == ["triggered"]
    assert stopped_events == ["stopped"]


@pytest.mark.asyncio
async def test_external_completion_strategy_can_ignore_legacy_user_stopped_frame():
    strategy = ExternalUserTurnCompletionStopStrategy(
        complete_on_user_stopped_speaking=False,
        enable_user_speaking_frames=False,
    )

    inference_events: list[str] = []
    stopped_events: list[str] = []

    @strategy.event_handler("on_user_turn_inference_triggered")
    async def on_user_turn_inference_triggered(strategy):
        inference_events.append("triggered")

    @strategy.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(strategy, params):
        stopped_events.append("stopped")

    assert await strategy.process_frame(UserStoppedSpeakingFrame()) == ProcessFrameResult.CONTINUE
    assert inference_events == []
    assert stopped_events == []
