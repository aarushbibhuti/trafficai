"""Opt-in GUI boundary; no calls to this module in headless processing."""

import os
import sys

import cv2


def require_display():
    if sys.platform.startswith("linux") and not (
        os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY")
    ):
        raise ValueError(
            "Preview needs a display. Omit --preview/--interactive on headless systems"
        )
    if "GUI:                           NONE" in cv2.getBuildInformation():
        raise ValueError("GUI unavailable: install the declared opencv-python dependency")


def show(frame):
    cv2.imshow("TrafficAI - Q to stop", frame)
    return cv2.waitKey(1) & 0xFF != ord("q")


def close():
    cv2.destroyAllWindows()
