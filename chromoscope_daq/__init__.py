"""Chromoscope DAQ.

Beam-profile data acquisition application for the Thorlabs CS165CU1/M
color camera, built on the Thorlabs TSI Scientific Camera SDK
(``thorlabs_tsi_sdk``).

The application operates in two modes, distinguished only by triggering:

- **Streaming** — free-run acquisition (no triggers), auto-exposure
  enabled by default.
- **Time Series** — software-triggered acquisition with fixed timing
  (Δt), manual exposure/gain required.
"""

__version__ = "0.1.0"
