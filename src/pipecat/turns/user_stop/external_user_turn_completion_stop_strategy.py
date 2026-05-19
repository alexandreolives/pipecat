#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""User turn stop strategy that supports eager inference and finalization frames."""

from pipecat.frames.frames import (
    Frame,
    UserStoppedSpeakingFrame,
    UserTurnInferenceCompletedFrame,
    UserTurnInferenceTriggeredFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy


class ExternalUserTurnCompletionStopStrategy(BaseUserTurnStopStrategy):
    """Drive eager inference and finalization from external turn-detection frames.

    Generic stop strategy for pipelines where some external component
    (STT with built-in turn detection, a dedicated end-of-turn classifier,
    custom user code, etc.) can emit a provisional inference trigger before
    the final completion signal arrives.

    The strategy accepts three frames:

    - :class:`~pipecat.frames.frames.UserTurnInferenceTriggeredFrame`
      starts inference only.
    - :class:`~pipecat.frames.frames.UserTurnInferenceCompletedFrame`
      starts inference if needed, then finalizes the turn.
    - :class:`~pipecat.frames.frames.UserStoppedSpeakingFrame` acts as a
      legacy finalization signal for pipelines that still emit the older
      stop frame.

    This lets a turn detector start the LLM early on a likely-final transcript
    and then confirm or cancel it later without changing the surrounding
    pipeline.

    Pair this with any start strategy that emits
    :class:`~pipecat.frames.frames.UserStartedSpeakingFrame`.
    For pipelines that still emit only ``UserStoppedSpeakingFrame``, this
    strategy preserves the old "fire inference and stop together" behavior.
    Example::

        stop=[ExternalUserTurnCompletionStopStrategy(enable_user_speaking_frames=False)]

    For LLM-completion-marker gating specifically, use the subclass
    :class:`~pipecat.turns.user_stop.LLMTurnCompletionUserTurnStopStrategy`
    instead, which additionally pushes the ``LLMUpdateSettingsFrame`` that
    enables the marker protocol on the LLM.

    If the producer never emits a completion frame, the controller's
    ``user_turn_stop_timeout`` watchdog finalizes the turn after no activity.
    Tune that timeout if your producer can take longer than the default to
    respond.
    """

    def __init__(self, *, complete_on_user_stopped_speaking: bool = True, **kwargs):
        """Initialize the external completion strategy.

        Args:
            complete_on_user_stopped_speaking: Treat ``UserStoppedSpeakingFrame`` as
                a legacy completion signal. Disable this when the producer emits
                explicit ``UserTurnInferenceCompletedFrame`` frames, otherwise a
                speaking-state frame can finalize an unconfirmed turn.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self._complete_on_user_stopped_speaking = complete_on_user_stopped_speaking
        self._inference_triggered = False
        self._turn_finalized = False

    async def reset(self):
        await super().reset()
        self._inference_triggered = False
        self._turn_finalized = False

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Handle inference-trigger and finalization frames from an external turn detector."""
        if isinstance(frame, UserTurnInferenceTriggeredFrame):
            await self._trigger_inference_once()
            return ProcessFrameResult.CONTINUE

        if isinstance(frame, UserTurnInferenceCompletedFrame):
            await self._trigger_completion()
            return ProcessFrameResult.STOP

        if isinstance(frame, UserStoppedSpeakingFrame) and self._complete_on_user_stopped_speaking:
            await self._trigger_completion()
            return ProcessFrameResult.STOP

        return ProcessFrameResult.CONTINUE

    async def mark_inference_triggered(self):
        self._inference_triggered = True

    async def _trigger_inference_once(self):
        if self._inference_triggered:
            return
        self._inference_triggered = True
        await self.trigger_user_turn_inference_triggered()

    async def _trigger_completion(self):
        if self._turn_finalized:
            return

        # Preserve the old behavior for legacy stop-frame emitters: if the
        # completion signal arrives before an explicit inference-trigger frame,
        # we still need to start inference once before finalizing.
        if not self._inference_triggered:
            await self._trigger_inference_once()

        self._turn_finalized = True
        await self.trigger_user_turn_finalized()
