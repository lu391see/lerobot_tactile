# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
"""
Provides the OpenCVCamera class for capturing frames from cameras using OpenCV
via Python 3.8+ SharedMemory.
"""

import logging
import math
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..utils import get_cv2_backend, get_cv2_rotation
from .configuration_opencv import ColorMode, OpenCVCameraConfig

MAX_OPENCV_INDEX = 60

logger = logging.getLogger(__name__)


def _mp_camera_worker(
    index_or_path: Any,
    backend: int,
    config: OpenCVCameraConfig,
    capture_width: int,
    capture_height: int,
    shm_name: str,
    new_frame_event: Any,
    stop_event: Any,
    ready_event: Any,
    error_queue: Any,
) -> None:
    """
    Background worker that strictly reads RAW frames from the hardware and dumps
    them into shared memory. All post-processing is left to the main process.
    """
    try:
        cv2.setNumThreads(1)
        cap = cv2.VideoCapture(index_or_path, backend)

        if not cap.isOpened():
            error_queue.put(f"Failed to open camera @ {index_or_path}")
            ready_event.set()
            return

        if config.fourcc is not None:
            fourcc_code = cv2.VideoWriter_fourcc(*config.fourcc)
            cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(capture_width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(capture_height))

        if config.fps is not None:
            cap.set(cv2.CAP_PROP_FPS, float(config.fps))

        # Hardware Validation
        actual_w = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_h = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if actual_w != capture_width or actual_h != capture_height:
            error_queue.put(f"Hardware rejected resolution. Requested {capture_width}x{capture_height}, got {actual_w}x{actual_h}")
            ready_event.set()
            return

        # Attach to shared memory (shape is strictly the RAW capture shape)
        shm = SharedMemory(name=shm_name)
        shared_np = np.ndarray((capture_height, capture_width, 3), dtype=np.uint8, buffer=shm.buf)

        # Worker Warmup: Guarantee the hardware actually streams the first frame
        ret, frame = cap.read()
        if not ret or frame is None:
            error_queue.put(f"Hardware opened {index_or_path}, but failed to grab the first frame.")
            ready_event.set()
            return

        np.copyto(shared_np, frame)
        new_frame_event.set()
        
        # Signal main process that hardware is streaming
        ready_event.set()

        # The lean, mean, raw read loop
        while not stop_event.is_set():
            ret, frame = cap.read()
            
            if not ret or frame is None:
                logger.warning(f"Hardware dropped a frame on {index_or_path}. Retrying...")
                time.sleep(0.01)
                continue

            np.copyto(shared_np, frame)
            new_frame_event.set()

    except Exception as e:
        import traceback
        error_msg = f"Worker Exception: {e}\n{traceback.format_exc()}"
        error_queue.put(error_msg)
        ready_event.set()
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        if 'shm' in locals() and shm is not None:
            shm.close()


class OpenCVCamera(Camera):
    """
    Manages camera interactions using OpenCV and SharedMemory.
    Operates fully asynchronously under the hood to prevent GIL delays.
    """

    def __init__(self, config: OpenCVCameraConfig):
        super().__init__(config)

        self.config = config
        self.index_or_path = config.index_or_path

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s

        self.rotation: int | None = get_cv2_rotation(config.rotation)
        self.backend: int = get_cv2_backend()

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width
        else:
            self.capture_width, self.capture_height = None, None

        # Multiprocessing primitives
        self.process: mp.Process | None = None
        self.mp_stop_event: Any = None
        self.mp_new_frame_event: Any = None
        self._shm: SharedMemory | None = None
        self._shared_np: NDArray[Any] | None = None
        self.error_queue: Any = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.index_or_path})"

    @property
    def is_connected(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def connect(self, warmup: bool = True) -> None:
        """
        Probes the camera (if needed), starts the background worker, and runs the warmup loop.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # Probe camera to figure out default dimensions if not explicitly provided
        needs_probe = self.capture_width is None or self.fps is None
        if needs_probe:
            cap = cv2.VideoCapture(self.index_or_path, self.backend)
            if not cap.isOpened():
                raise ConnectionError(f"Failed to open {self}.")
            
            if self.capture_width is None:
                default_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                default_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.capture_width, self.capture_height = default_w, default_h
                self.width, self.height = default_w, default_h
                if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                    self.width, self.height = default_h, default_w
                    self.capture_width, self.capture_height = default_w, default_h
            
            if self.fps is None:
                self.fps = cap.get(cv2.CAP_PROP_FPS)
            
            cap.release()
            time.sleep(0.5) # Allow USB to reset before worker grabs it

        # Start the worker process
        self._start_read_process()

        # Warmup loop (reading from the worker)
        if warmup:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.read()
                time.sleep(0.01)

        logger.info(f"{self} connected natively via Multiprocessing.")

    def _start_read_process(self) -> None:
        ctx = mp.get_context('spawn')
        
        # Buffer shape must be raw shape before rotation
        array_size = self.capture_height * self.capture_width * 3
        self._shm = SharedMemory(create=True, size=array_size)
        self._shared_np = np.ndarray((self.capture_height, self.capture_width, 3), dtype=np.uint8, buffer=self._shm.buf)

        self.mp_stop_event = ctx.Event()
        self.mp_new_frame_event = ctx.Event()
        ready_event = ctx.Event()
        self.error_queue = ctx.Queue()

        self.process = ctx.Process(
            target=_mp_camera_worker,
            args=(
                self.index_or_path, self.backend, self.config,
                self.capture_width, self.capture_height,
                self._shm.name, self.mp_new_frame_event, 
                self.mp_stop_event, ready_event, self.error_queue
            ),
            daemon=True,
            name=f"{self}_mp_worker"
        )
        self.process.start()

        if not ready_event.wait(timeout=15.0):
            self.disconnect()
            raise ConnectionError(f"Timeout waiting for {self} process to initialize.")
            
        if not self.error_queue.empty():
            self.disconnect()
            raise ConnectionError(f"Camera worker failed: {self.error_queue.get()}")

    def _stop_read_process(self) -> None:
        if self.mp_stop_event is not None:
            self.mp_stop_event.set()

        if self.process is not None and self.process.is_alive():
            self.process.join(timeout=2.0)
            if self.process.is_alive():
                logger.warning(f"Force terminating {self} worker process.")
                self.process.terminate()
                self.process.join()

        self.process = None
        self.mp_stop_event = None

        if self._shm is not None:
            self._shm.close()
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            self._shm = None
        self._shared_np = None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        # ... [Unchanged find_cameras logic] ...
        found_cameras_info = []
        targets_to_scan: list[str | int]
        if platform.system() == "Linux":
            possible_paths = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
            targets_to_scan = [str(p) for p in possible_paths]
        else:
            targets_to_scan = [int(i) for i in range(MAX_OPENCV_INDEX)]

        for target in targets_to_scan:
            camera = cv2.VideoCapture(target)
            if camera.isOpened():
                default_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                default_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                default_fps = camera.get(cv2.CAP_PROP_FPS)
                default_format = camera.get(cv2.CAP_PROP_FORMAT)
                default_fourcc_code = camera.get(cv2.CAP_PROP_FOURCC)
                default_fourcc_code_int = int(default_fourcc_code)
                default_fourcc = "".join([chr((default_fourcc_code_int >> 8 * i) & 0xFF) for i in range(4)])
                camera_info = {
                    "name": f"OpenCV Camera @ {target}",
                    "type": "OpenCV",
                    "id": target,
                    "backend_api": camera.getBackendName(),
                    "default_stream_profile": {
                        "format": default_format,
                        "fourcc": default_fourcc,
                        "width": default_width,
                        "height": default_height,
                        "fps": default_fps,
                    },
                }
                found_cameras_info.append(camera_info)
                camera.release()
        return found_cameras_info

    def read(self, color_mode: ColorMode | None = None) -> NDArray[Any]:
        """Reads a new frame synchronously from the background worker."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        start_time = time.perf_counter()

        # Clear event to force waiting for a brand new hardware frame
        self.mp_new_frame_event.clear()
        if not self.mp_new_frame_event.wait(timeout=2.0):
            if hasattr(self, 'error_queue') and not self.error_queue.empty():
                raise RuntimeError(f"Worker crashed: {self.error_queue.get()}")
            raise RuntimeError(f"{self} read failed: Timeout.")

        raw_frame = self._shared_np.copy()
        processed_frame = self._postprocess_image(raw_frame, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")
        
        return processed_frame

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Reads the latest available frame asynchronously."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if not self.mp_new_frame_event.wait(timeout=timeout_ms / 1000.0):
            if hasattr(self, 'error_queue') and not self.error_queue.empty():
                raise RuntimeError(f"Worker crashed: {self.error_queue.get()}")
            raise TimeoutError(f"Timed out waiting for frame from camera {self}.")

        raw_frame = self._shared_np.copy()
        self.mp_new_frame_event.clear()
        
        return self._postprocess_image(raw_frame)

    def _postprocess_image(self, image: NDArray[Any], color_mode: ColorMode | None = None) -> NDArray[Any]:
        """Applies color conversion and rotation in the main process."""
        requested_color_mode = self.color_mode if color_mode is None else color_mode
        if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(f"Invalid color mode '{requested_color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}.")
        
        # We validate against capture_height/capture_width (the raw dimensions)
        h, w, c = image.shape
        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}.")
        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")
        
        processed_image = image
        if requested_color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)
            
        return processed_image

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} not connected.")

        self._stop_read_process()
        logger.info(f"{self} disconnected.")