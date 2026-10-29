"""Review overlays; missing speed is displayed explicitly."""

import cv2

from .calibration import draw_calibration
from .config import display_speed


def label_text(frame, text, x, y, color):
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.45
    (width, height), baseline = cv2.getTextSize(text, font, scale, 1)
    x = max(0, min(x, frame.shape[1] - width - 4))
    y = max(height + 4, min(y, frame.shape[0] - baseline - 2))
    cv2.rectangle(frame, (x, y - height - 3), (x + width + 4, y + baseline), (15, 15, 15), -1)
    cv2.putText(frame, text, (x + 2, y), font, scale, color, 1, cv2.LINE_AA)


def annotate(frame, observations, timestamp, calibration, features, units, active):
    output = draw_calibration(frame, calibration) if calibration else frame.copy()
    features.draw(output)
    for detection, measurement in observations:
        x1, y1, x2, y2 = map(int, detection.bbox)
        event = active.get(detection.track_id)
        confirmed = event is not None and event.confirmed_seconds is not None
        color = (40, 40, 240) if confirmed else (50, 210, 80)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = (
            f"#{detection.track_id} {detection.label} {detection.confidence:.2f} "
            f"{display_speed(measurement.speed_mps, units)}"
        )
        if confirmed:
            label += f" CANDIDATE {event.kind}"
        label_text(output, label, x1, y1 - 8, color)
        cv2.circle(output, tuple(map(int, detection.contact)), 3, color, -1)
    label_text(
        output,
        f"t={timestamp:.3f}s | candidate events require review",
        10,
        24,
        (255, 255, 255),
    )
    return output
