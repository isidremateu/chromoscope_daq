"""Output writers: MP4 (dedicated thread, drop counter) and TIFF stacks."""

from __future__ import annotations

import queue
import threading
from typing import Optional

import cv2
import numpy as np


class Mp4Writer:
    """Bounded-queue mp4 writer running in its own thread.

    ``write(bgr8)`` never blocks: when the queue is full the frame is
    dropped and counted (``dropped``), so acquisition/display pacing is
    never stalled by slow disk I/O (README: saving prioritized over
    display, drop counter when I/O is slower than the fps).
    """

    def __init__(self, path: str, fps: float, size: tuple,  # (width, height)
                 queue_depth: int = 30) -> None:
        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=queue_depth)
        self._sentinel: Optional[np.ndarray] = None
        self._writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (int(size[0]), int(size[1])))
        self._dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="mp4-writer")
        self._thread.start()

    @property
    def dropped(self) -> int:
        return self._dropped

    def write(self, frame_bgr8: np.ndarray) -> None:
        try:
            self._queue.put_nowait(frame_bgr8)
        except queue.Full:
            self._dropped += 1

    def close(self) -> None:
        """Flush the queue, stop the writer thread, release the file."""
        self._queue.put(self._sentinel)
        self._thread.join()
        self._writer.release()

    def _run(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:  # sentinel
                # Drain anything queued behind the sentinel is not needed;
                # the sentinel is only enqueued by close().
                return
            self._writer.write(frame)


class TiffStackWriter:
    """Append frames to a multi-page TIFF file (classic or OME)."""

    def __init__(self, path: str, ome: bool = False) -> None:
        import tifffile

        self._path = path
        self._frames = 0
        # tifffile: ome=True writes an OME-TIFF; bigtiff off is fine for
        # 16-bit frames unless the stack exceeds 4 GB (user can stop before).
        self._writer = tifffile.TiffWriter(path, ome=ome,
                                           bigtiff=False, mode="w")

    @property
    def frames(self) -> int:
        return self._frames

    def write(self, frame16: np.ndarray) -> None:
        self._writer.write(frame16, contiguous=False)
        self._frames += 1

    def close(self) -> None:
        self._writer.close()
