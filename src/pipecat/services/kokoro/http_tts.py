#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Kokoro-FastAPI text-to-speech service implementation."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

import aiohttp
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame
from pipecat.services.settings import NOT_GIVEN, TTSSettings, _NotGiven, assert_given
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts


@dataclass
class KokoroHttpTTSSettings(TTSSettings):
    """Settings for KokoroHttpTTSService."""

    speed: float | None | _NotGiven = field(default_factory=lambda: NOT_GIVEN)


class KokoroHttpTTSService(TTSService):
    """Kokoro TTS service using the Dockerized FastAPI wrapper."""

    Settings = KokoroHttpTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8300",
        aiohttp_session: aiohttp.ClientSession | None = None,
        api_key: str | None = None,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Kokoro HTTP TTS service.

        Args:
            base_url: Kokoro-FastAPI base URL. The service appends
                ``/api/tts/stream`` automatically unless already provided.
            aiohttp_session: Optional shared aiohttp session.
            api_key: Reserved for future use. The self-hosted image does not
                require authentication.
            settings: Runtime-updatable settings.
            **kwargs: Additional keyword arguments passed to ``TTSService``.
        """
        default_settings = self.Settings(
            model="kokoro",
            voice="ff_siwis",
            language=None,
            speed=1.0,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )

        normalized_base_url = base_url.rstrip("/")
        if normalized_base_url.endswith("/v1"):
            # Accept the legacy OpenAI-style base URL so existing configs can
            # migrate without needing a manual URL rewrite.
            normalized_base_url = normalized_base_url[: -len("/v1")]
        if normalized_base_url.endswith("/api/tts/stream"):
            self._tts_url = normalized_base_url
        elif normalized_base_url.endswith("/api/tts"):
            self._tts_url = f"{normalized_base_url}/stream"
        else:
            self._tts_url = f"{normalized_base_url}/api/tts/stream"

        self._api_key = api_key
        self._session = aiohttp_session
        self._session_owner = aiohttp_session is None

    def can_generate_metrics(self) -> bool:
        """Check whether this service can generate processing metrics."""
        return True

    async def start(self, frame):
        """Start the Kokoro TTS service."""
        await super().start(frame)
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

    async def stop(self, frame):
        """Stop the Kokoro TTS service."""
        await super().stop(frame)
        await self._close_session()

    async def cancel(self, frame):
        """Cancel the Kokoro TTS service."""
        await super().cancel(frame)
        await self._close_session()

    async def _close_session(self):
        if self._session_owner and self._session and not self._session.closed:
            await self._session.close()
        if self._session_owner:
            self._session = None

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Generate speech from text using Kokoro-FastAPI."""
        logger.debug(f"{self}: Generating TTS [{text}]")

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._session_owner = True

        voice = assert_given(self._settings.voice)
        speed = assert_given(self._settings.speed)
        if voice is None:
            yield ErrorFrame(error="Kokoro TTS voice must be specified")
            return
        if speed is None:
            yield ErrorFrame(error="Kokoro TTS speed must be specified")
            return

        payload = {
            "text": text,
            "voice": voice,
            "speed": speed,
        }
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        measuring_ttfb = True
        try:
            await self.start_tts_usage_metrics(text)
            timeout = aiohttp.ClientTimeout(total=120)
            async with self._session.post(
                self._tts_url,
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as response:
                if response.status != 200:
                    error = await response.text(errors="ignore")
                    logger.error(
                        f"{self} error getting audio (status: {response.status}, error: {error})"
                    )
                    yield ErrorFrame(
                        error=f"Error getting audio (status: {response.status}, error: {error})"
                    )
                    return

                async for frame in self._stream_audio_frames_from_iterator(
                    response.content.iter_chunked(self.chunk_size),
                    strip_wav_header=True,
                    context_id=context_id,
                ):
                    if measuring_ttfb:
                        await self.stop_ttfb_metrics()
                        measuring_ttfb = False
                    yield frame
        except Exception as e:
            yield ErrorFrame(error=f"Unknown error occurred: {e}")
        finally:
            await self.stop_ttfb_metrics()
