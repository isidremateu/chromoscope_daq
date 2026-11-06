"""Acquisition modes for Chromoscope DAQ.

Each mode is a session class taking the open camera plus user settings,
runnable once (``run``) and returning a stats object. File naming follows
``YYMMDD_HHMMSS_identifier.ext`` with the timestamp fixed at start.
"""

from .base import AcquisitionError, FrameBundle, Processor, poll_frame_copy
from .naming import acquisition_timestamp, build_basename, build_filename
from .autoexposure import AutoExposureController
from .processing import COLOR_SPACES, default_processor, make_processor
from .writers import Mp4Writer, TiffStackWriter
from .snapshot import SnapshotSession, save_snapshot
from .streaming import StreamingSession, StreamingStats
from .time_series import (TimeSeriesSession, TimeSeriesStats,
                          camera_fps_max, minimum_interval_s)

__all__ = [
    "AcquisitionError",
    "FrameBundle",
    "Processor",
    "poll_frame_copy",
    "COLOR_SPACES",
    "default_processor",
    "make_processor",
    "acquisition_timestamp",
    "build_basename",
    "build_filename",
    "AutoExposureController",
    "Mp4Writer",
    "TiffStackWriter",
    "SnapshotSession",
    "save_snapshot",
    "StreamingSession",
    "StreamingStats",
    "TimeSeriesSession",
    "TimeSeriesStats",
    "camera_fps_max",
    "minimum_interval_s",
]
