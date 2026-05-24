import asyncio
import io
import wave
from contextlib import asynccontextmanager

import aiohttp
import pytest
from aiohttp import web

from pipecat.frames.frames import (
    AggregatedTextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.kokoro.http_tts import KokoroHttpTTSService, KokoroHttpTTSSettings
from pipecat.tests.utils import run_test


def build_wav_bytes(sample_rate: int = 24000) -> tuple[bytes, bytes]:
    pcm = b"\x01\x02" * 2400
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue(), pcm


@asynccontextmanager
async def run_test_server(app: web.Application, host: str, port: int):
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    try:
        yield
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_run_kokoro_http_tts_streams_wav_audio(unused_tcp_port):
    request_bodies = []
    wav_bytes, pcm_bytes = build_wav_bytes()

    async def handler(request):
        request_bodies.append(await request.json())

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={"Content-Type": "audio/wav"},
        )
        await response.prepare(request)
        await response.write(wav_bytes)
        await asyncio.sleep(0.01)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/api/tts/stream", handler)
    host = "127.0.0.1"
    port = unused_tcp_port
    base_url = f"http://{host}:{port}"

    async with run_test_server(app, host, port):
        async with aiohttp.ClientSession() as session:
            tts_service = KokoroHttpTTSService(
                base_url=base_url,
                aiohttp_session=session,
                sample_rate=24000,
                settings=KokoroHttpTTSSettings(
                    model="kokoro",
                    voice="ff_siwis",
                    speed=1.0,
                ),
            )

            down_frames, _ = await run_test(
                tts_service,
                frames_to_send=[TTSSpeakFrame(text="Bonjour le monde.")],
            )

    frame_types = [type(frame) for frame in down_frames]
    assert AggregatedTextFrame in frame_types
    assert TTSStartedFrame in frame_types
    assert TTSStoppedFrame in frame_types
    assert TTSTextFrame in frame_types

    audio_frames = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    assert audio_frames
    assert all(frame.sample_rate == 24000 for frame in audio_frames)
    assert all(frame.num_channels == 1 for frame in audio_frames)
    assert b"".join(frame.audio for frame in audio_frames) == pcm_bytes

    assert request_bodies == [
        {
            "text": "Bonjour le monde.",
            "voice": "ff_siwis",
            "speed": 1.0,
        }
    ]


@pytest.mark.asyncio
async def test_run_kokoro_http_tts_accepts_legacy_v1_base_url(unused_tcp_port):
    request_bodies = []
    wav_bytes, _ = build_wav_bytes()

    async def handler(request):
        request_bodies.append(await request.json())

        response = web.StreamResponse(
            status=200,
            reason="OK",
            headers={"Content-Type": "audio/wav"},
        )
        await response.prepare(request)
        await response.write(wav_bytes)
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/api/tts/stream", handler)
    host = "127.0.0.1"
    port = unused_tcp_port
    base_url = f"http://{host}:{port}/v1"

    async with run_test_server(app, host, port):
        async with aiohttp.ClientSession() as session:
            tts_service = KokoroHttpTTSService(
                base_url=base_url,
                aiohttp_session=session,
                sample_rate=24000,
                settings=KokoroHttpTTSSettings(
                    model="kokoro",
                    voice="ff_siwis",
                    speed=1.0,
                ),
            )

            down_frames, _ = await run_test(
                tts_service,
                frames_to_send=[TTSSpeakFrame(text="Bonjour le monde.")],
            )

    assert request_bodies == [
        {
            "text": "Bonjour le monde.",
            "voice": "ff_siwis",
            "speed": 1.0,
        }
    ]
    audio_frames = [frame for frame in down_frames if isinstance(frame, TTSAudioRawFrame)]
    assert audio_frames
