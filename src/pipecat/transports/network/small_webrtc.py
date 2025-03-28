#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import fractions
import time
from collections import deque
from typing import Any, Awaitable, Callable, Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InputImageRawFrame,
    OutputImageRawFrame,
    StartFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.webrtc_connection import SmallWebRTCConnection

try:
    import cv2
    from aiortc import VideoStreamTrack
    from aiortc.mediastreams import AudioStreamTrack, MediaStreamError
    from av import AudioFrame, AudioResampler, VideoFrame
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use the SmallWebRTC, you need to `pip install pipecat-ai[webrtc]`.")
    raise Exception(f"Missing module: {e}")


class SmallWebRTCCallbacks(BaseModel):
    on_app_message: Callable[[Any], Awaitable[None]]
    on_client_connected: Callable[[SmallWebRTCConnection], Awaitable[None]]
    on_client_disconnected: Callable[[SmallWebRTCConnection], Awaitable[None]]
    on_client_closed: Callable[[SmallWebRTCConnection], Awaitable[None]]


class RawAudioTrack(AudioStreamTrack):
    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self._samples_per_frame = self._sample_rate // 50  # 20ms per frame
        self._timestamp = 0
        self._audio_buffer = deque()
        self._start = time.time()

    def add_audio_bytes(self, audio_bytes: bytes):
        """
        Adds bytes to the audio buffer and returns a Future that completes when the data is processed.
        """
        if len(audio_bytes) % 2 != 0:
            raise ValueError("Audio bytes length must be even (16-bit samples).")
        future = asyncio.get_running_loop().create_future()
        self._audio_buffer.append((audio_bytes, future))
        return future

    async def recv(self):
        """
        Returns the next audio frame, generating silence if needed.
        """
        # Compute required wait time for synchronization
        if self._timestamp > 0:
            wait = self._start + (self._timestamp / self._sample_rate) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

        # Check if we have enough data
        needed_bytes = self._samples_per_frame * 2  # 16-bit (2 bytes per sample)
        available_bytes = sum(len(audio_bytes) for audio_bytes, _ in self._audio_buffer)
        consumed_futures = []  # Track futures for processed data
        if available_bytes >= needed_bytes:
            # Extract data from deque
            chunk = bytearray()
            while len(chunk) < needed_bytes:
                audio_bytes, future = self._audio_buffer.popleft()
                chunk.extend(audio_bytes)
                consumed_futures.append(future)  # Track the future
            chunk = bytes(chunk[:needed_bytes])  # Trim excess bytes
        else:
            chunk = bytes(needed_bytes)  # Generate silent frame

        # Convert the byte data to an ndarray of int16 samples
        samples = np.frombuffer(chunk, dtype=np.int16)

        # Create AudioFrame
        frame = AudioFrame.from_ndarray(samples[None, :], layout="mono")
        self._timestamp += self._samples_per_frame
        frame.pts = self._timestamp
        frame.sample_rate = self._sample_rate
        frame.time_base = fractions.Fraction(1, self._sample_rate)

        # Resolve all futures corresponding to consumed data
        for future in consumed_futures:
            if not future.done():
                future.set_result(True)

        return frame


class RawVideoTrack(VideoStreamTrack):
    def __init__(self, width, height):
        super().__init__()
        self._width = width
        self._height = height
        self._video_buffer = asyncio.Queue()

    def add_video_frame(self, frame):
        """Adds a raw video frame to the buffer."""
        self._video_buffer.put_nowait(frame)

    async def recv(self):
        """Returns the next video frame, waiting if the buffer is empty."""
        raw_frame = await self._video_buffer.get()

        # Convert bytes to NumPy array
        frame_data = np.frombuffer(raw_frame.image, dtype=np.uint8).reshape(
            (self._height, self._width, 3)
        )

        frame = VideoFrame.from_ndarray(frame_data, format="rgb24")

        # Assign timestamp
        frame.pts, frame.time_base = await self.next_timestamp()

        return frame


class SmallWebRTCClient:
    def __init__(self, webrtc_connection: SmallWebRTCConnection, callbacks: SmallWebRTCCallbacks):
        self._webrtc_connection = webrtc_connection
        self._closing = False
        self._callbacks = callbacks

        self._audio_output_track = None
        self._video_output_track = None
        self._audio_input_track: Optional[AudioStreamTrack] = None
        self._video_input_track: Optional[VideoStreamTrack] = None

        self._params = None
        self._audio_in_channels = None
        self._in_sample_rate = None
        self._out_sample_rate = None

        # We are always resampling it for 16000 if the sample_rate that we receive is bigger than that.
        # otherwise we face issues with Silero VAD
        self._pipecat_resampler = AudioResampler("s16", "mono", 16000)

        @self._webrtc_connection.event_handler("connected")
        async def on_connected(connection: SmallWebRTCConnection):
            logger.debug("Peer connection established.")
            await self._handle_client_connected()

        @self._webrtc_connection.event_handler("disconnected")
        async def on_disconnected(connection: SmallWebRTCConnection):
            logger.debug("Peer connection lost.")
            await self._handle_client_disconnected()

        @self._webrtc_connection.event_handler("closed")
        async def on_closed(connection: SmallWebRTCConnection):
            logger.debug("Client connection closed.")
            await self._handle_client_closed()

        @self._webrtc_connection.event_handler("app-message")
        async def on_app_message(connection: SmallWebRTCConnection, message: Any):
            await self._handle_app_message(message)

    async def read_video_frame(self):
        """
        Reads a video frame from the given MediaStreamTrack, converts it to RGB,
        and creates an InputImageRawFrame.
        """
        while True:
            if self._video_input_track is None:
                await asyncio.sleep(0.01)
                continue

            try:
                frame = await asyncio.wait_for(self._video_input_track.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                if self._webrtc_connection.is_connected():
                    logger.warning("Timeout: No video frame received within the specified time.")
                    # self._webrtc_connection.ask_to_renegotiate()
                frame = None
            except MediaStreamError:
                logger.warning("Received an unexpected media stream error while reading the audio.")
                frame = None

            if frame is None or not isinstance(frame, VideoFrame):
                # If no valid frame, sleep for a bit
                await asyncio.sleep(0.01)
                continue

            format_name = frame.format.name

            # Convert frame to NumPy array in its native format
            frame_array = frame.to_ndarray(format=format_name)

            # Handle different formats dynamically
            if format_name == "yuv420p":
                frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_YUV2RGB_I420)
            elif format_name == "nv12":
                frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_YUV2RGB_NV12)
            elif format_name == "gray":
                frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_GRAY2RGB)
            elif format_name.startswith("rgb"):  # Already RGB, no conversion needed
                frame_rgb = frame_array
            else:
                raise ValueError(f"Unsupported format: {format_name}")

            image_frame = InputImageRawFrame(
                image=frame_rgb.tobytes(),
                size=(frame.width, frame.height),
                format="RGB",
            )

            yield image_frame

    async def read_audio_frame(self):
        """
        Reads 20ms of audio from the given MediaStreamTrack and creates an InputAudioRawFrame.
        """
        while True:
            if self._audio_input_track is None:
                await asyncio.sleep(0.01)
                continue

            try:
                frame = await asyncio.wait_for(self._audio_input_track.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                if self._webrtc_connection.is_connected():
                    logger.warning("Timeout: No audio frame received within the specified time.")
                frame = None
            except MediaStreamError:
                logger.warning("Received an unexpected media stream error while reading the audio.")
                frame = None

            if frame is None or not isinstance(frame, AudioFrame):
                # If we don't read any audio let's sleep for a little bit (i.e. busy wait).
                await asyncio.sleep(0.01)
                continue

            if frame.sample_rate > self._in_sample_rate:
                resampled_frames = self._pipecat_resampler.resample(frame)
                for resampled_frame in resampled_frames:
                    # 16-bit PCM bytes
                    pcm_bytes = resampled_frame.to_ndarray().astype(np.int16).tobytes()
                    audio_frame = InputAudioRawFrame(
                        audio=pcm_bytes,
                        sample_rate=resampled_frame.sample_rate,
                        num_channels=self._audio_in_channels,
                    )
                    yield audio_frame
            else:
                # 16-bit PCM bytes
                pcm_bytes = frame.to_ndarray().astype(np.int16).tobytes()
                audio_frame = InputAudioRawFrame(
                    audio=pcm_bytes,
                    sample_rate=frame.sample_rate,
                    num_channels=self._audio_in_channels,
                )
                yield audio_frame

    async def write_raw_audio_frames(self, data: bytes):
        if self._can_send() and self._audio_output_track:
            await self._audio_output_track.add_audio_bytes(data)

    async def write_frame_to_camera(self, frame: OutputImageRawFrame):
        if self._can_send() and self._video_output_track:
            self._video_output_track.add_video_frame(frame)

    async def setup(self, _params: TransportParams, frame):
        self._audio_in_channels = _params.audio_in_channels
        self._in_sample_rate = _params.audio_in_sample_rate or frame.audio_in_sample_rate
        self._out_sample_rate = _params.audio_out_sample_rate or frame.audio_out_sample_rate
        self._params = _params

    async def connect(self):
        if self._webrtc_connection.is_connected():
            # already initialized
            return

        logger.info(f"Connecting to Small WebRTC")
        await self._webrtc_connection.connect()

    async def disconnect(self):
        if self.is_connected and not self.is_closing:
            logger.info(f"Disconnecting to Small WebRTC")
            self._closing = True
            await self._webrtc_connection.close()
            await self._handle_client_disconnected()

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        if self._can_send():
            self._webrtc_connection.send_app_message(frame.message)

    async def _handle_client_connected(self):
        # There is nothing to do here yet, the pipeline is still not ready
        if not self._params:
            return

        self._audio_input_track = self._webrtc_connection.audio_input_track()
        self._video_input_track = self._webrtc_connection.video_input_track()
        if self._params.audio_out_enabled:
            self._audio_output_track = RawAudioTrack(sample_rate=self._out_sample_rate)
            self._webrtc_connection.replace_audio_track(self._audio_output_track)

        if self._params.camera_out_enabled:
            self._video_output_track = RawVideoTrack(
                width=self._params.camera_out_width, height=self._params.camera_out_height
            )
            self._webrtc_connection.replace_video_track(self._video_output_track)

        await self._callbacks.on_client_connected(self._webrtc_connection)

    async def _handle_client_disconnected(self):
        self._audio_input_track = None
        self._video_input_track = None
        self._audio_output_track = None
        self._video_output_track = None
        await self._callbacks.on_client_disconnected(self._webrtc_connection)

    async def _handle_client_closed(self):
        self._audio_input_track = None
        self._video_input_track = None
        self._audio_output_track = None
        self._video_output_track = None
        await self._callbacks.on_client_closed(self._webrtc_connection)

    async def _handle_app_message(self, message: Any):
        await self._callbacks.on_app_message(message)

    def _can_send(self):
        return self.is_connected and not self.is_closing

    @property
    def is_connected(self) -> bool:
        return self._webrtc_connection.is_connected()

    @property
    def is_closing(self) -> bool:
        return self._closing


class SmallWebRTCInputTransport(BaseInputTransport):
    def __init__(
        self,
        client: SmallWebRTCClient,
        params: TransportParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params
        self._receive_audio_task = None
        self._receive_video_task = None

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._client.setup(self._params, frame)
        await self._client.connect()
        if not self._receive_audio_task and (
            self._params.audio_in_enabled or self._params.vad_enabled
        ):
            self._receive_audio_task = self.create_task(self._receive_audio())
        if not self._receive_video_task and self._params.camera_in_enabled:
            self._receive_video_task = self.create_task(self._receive_video())

    async def _stop_tasks(self):
        if self._receive_audio_task:
            await self.cancel_task(self._receive_audio_task)
            self._receive_audio_task = None
        if self._receive_video_task:
            await self.cancel_task(self._receive_video_task)
            self._receive_video_task = None

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._stop_tasks()
        await self._client.disconnect()

    async def _receive_audio(self):
        try:
            async for audio_frame in self._client.read_audio_frame():
                if audio_frame:
                    await self.push_audio_frame(audio_frame)

        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

    async def _receive_video(self):
        try:
            async for video_frame in self._client.read_video_frame():
                if video_frame:
                    await self.push_frame(video_frame)

        except Exception as e:
            logger.error(f"{self} exception receiving data: {e.__class__.__name__} ({e})")

    async def push_app_message(self, message: Any):
        logger.debug(f"Received app message inside SmallWebRTCInputTransport  {message}")
        frame = TransportMessageUrgentFrame(message=message)
        await self.push_frame(frame)


class SmallWebRTCOutputTransport(BaseOutputTransport):
    def __init__(
        self,
        client: SmallWebRTCClient,
        params: TransportParams,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._client = client
        self._params = params

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._client.setup(self._params, frame)
        await self._client.connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._client.disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._client.disconnect()

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        await self._client.send_message(frame)

    async def write_raw_audio_frames(self, frames: bytes):
        await self._client.write_raw_audio_frames(frames)

    async def write_frame_to_camera(self, frame: OutputImageRawFrame):
        await self._client.write_frame_to_camera(frame)


class SmallWebRTCTransport(BaseTransport):
    def __init__(
        self,
        webrtc_connection: SmallWebRTCConnection,
        params: TransportParams,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)
        self._params = params

        self._callbacks = SmallWebRTCCallbacks(
            on_app_message=self._on_app_message,
            on_client_connected=self._on_client_connected,
            on_client_disconnected=self._on_client_disconnected,
            on_client_closed=self._on_client_closed,
        )

        self._client = SmallWebRTCClient(webrtc_connection, self._callbacks)

        self._input = SmallWebRTCInputTransport(self._client, self._params, name=self._input_name)
        self._output = SmallWebRTCOutputTransport(
            self._client, self._params, name=self._output_name
        )

        # Register supported handlers. The user will only be able to register
        # these handlers.
        self._register_event_handler("on_app_message")
        self._register_event_handler("on_client_connected")
        self._register_event_handler("on_client_disconnected")
        self._register_event_handler("on_client_closed")

    def input(self) -> SmallWebRTCInputTransport:
        if not self._input:
            self._input = SmallWebRTCInputTransport(
                self._client, self._params, name=self._input_name
            )
        return self._input

    def output(self) -> SmallWebRTCOutputTransport:
        if not self._output:
            self._output = SmallWebRTCOutputTransport(
                self._client, self._params, name=self._input_name
            )
        return self._output

    async def _on_app_message(self, message: Any):
        if self._input:
            await self._input.push_app_message(message)
        await self._call_event_handler("on_app_message", message)

    async def _on_client_connected(self, webrtc_connection):
        await self._call_event_handler("on_client_connected", webrtc_connection)

    async def _on_client_disconnected(self, webrtc_connection):
        await self._call_event_handler("on_client_disconnected", webrtc_connection)

    async def _on_client_closed(self, webrtc_connection):
        await self._call_event_handler("on_client_closed", webrtc_connection)
