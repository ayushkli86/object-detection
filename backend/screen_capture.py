"""
Screen Capture Module - Captures screen content for object detection.
Uses mss (Multi-Screen Shot) for fast, cross-platform screen capture.
"""

import mss
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class MonitorInfo:
    """Information about a connected monitor."""
    id: int
    left: int
    top: int
    width: int
    height: int
    name: str


@dataclass
class CaptureRegion:
    """Region to capture on screen."""
    left: int
    top: int
    width: int
    height: int


class ScreenCapture:
    """
    High-performance screen capture using MSS.
    Supports multi-monitor setups and custom capture regions.
    """

    def __init__(self, monitor_id: int = 0):
        """
        Initialize screen capture.

        Args:
            monitor_id: Monitor index to capture (0 = primary/full screen)
        """
        self.monitor_id = monitor_id
        self._sct = mss.mss()
        self._region: Optional[CaptureRegion] = None
        self._capture_ms = 0

        # Discover monitors
        self._refresh_monitors()

    def _refresh_monitors(self):
        """Refresh the list of available monitors."""
        self.monitors = []
        try:
            for i, mon in enumerate(self._sct.monitors):
                if i == 0:
                    self.monitors.append(MonitorInfo(
                        id=i, left=mon["left"], top=mon["top"],
                        width=mon["width"], height=mon["height"],
                        name="All-in-One"
                    ))
                else:
                    self.monitors.append(MonitorInfo(
                        id=i, left=mon["left"], top=mon["top"],
                        width=mon["width"], height=mon["height"],
                        name=f"Monitor {i}"
                    ))
        except Exception as e:
            logger.error(f"Failed to enumerate monitors: {e}")
            # Fallback: assume single 1920x1080 display
            self.monitors = [MonitorInfo(0, 0, 0, 1920, 1080, "Default")]

        logger.info(f"Found {len(self.monitors)} monitor(s)")

    def get_monitors(self) -> List[MonitorInfo]:
        """Get list of available monitors."""
        return self.monitors

    def set_region(self, region: Optional[CaptureRegion] = None):
        """
        Set a custom capture region.

        Args:
            region: Capture region or None for full monitor
        """
        self._region = region

    def set_monitor(self, monitor_id: int):
        """Switch to a different monitor."""
        if 0 <= monitor_id < len(self.monitors):
            self.monitor_id = monitor_id
            self._region = None  # Reset region for new monitor
            logger.info(f"Switched to monitor {monitor_id}")
        else:
            logger.warning(f"Monitor {monitor_id} not found, keeping current")

    def capture(self) -> Tuple[np.ndarray, float]:
        """
        Capture the current screen content.

        Returns:
            Tuple of (RGB numpy array, capture time in ms)
        """
        cap_start = time.time()

        try:
            if self._region:
                # Capture custom region
                region = {
                    "left": self._region.left,
                    "top": self._region.top,
                    "width": self._region.width,
                    "height": self._region.height,
                }
            else:
                # Capture selected monitor
                if self.monitor_id < len(self._sct.monitors):
                    region = self._sct.monitors[self.monitor_id]
                else:
                    region = self._sct.monitors[0]

            screenshot = self._sct.grab(region)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            frame = np.array(img)

            self._capture_ms = (time.time() - cap_start) * 1000
            return frame, self._capture_ms

        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            # Return a blank frame on error
            blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
            self._capture_ms = (time.time() - cap_start) * 1000
            return blank, self._capture_ms

    def cleanup(self):
        """Release resources."""
        if self._sct:
            self._sct.close()
        logger.info("Screen capture resources released")
