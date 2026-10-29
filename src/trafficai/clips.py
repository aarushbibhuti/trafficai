"""Seek and decode one annotated event clip at a time; bounded RAM and no frame cache."""

import av

from .video import VideoWriter


def make_clip(source, destination, start, end, width, height, fps):
    first = last = None
    count = 0
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        container.seek(int(start / stream.time_base), stream=stream, backward=True)
        with VideoWriter(destination, width, height, fps) as writer:
            for frame in container.decode(stream):
                timestamp = float(frame.pts * frame.time_base)
                if timestamp < start - 1e-6:
                    continue
                if timestamp > end + 1e-6:
                    break
                if first is None:
                    first = timestamp
                last = timestamp
                writer.write(frame.to_ndarray(format="bgr24"), timestamp - first)
                count += 1
    if not count:
        raise ValueError("No frames decoded for event clip")
    return {
        "path": f"clips/{destination.name}",
        "start_seconds": first,
        "end_seconds": last,
        "frames": count,
        "audio": False,
    }


def export_clips(exports, video_info, duration, config):
    directory = exports.directory / "clips"
    directory.mkdir(exist_ok=True)
    for event in exports.events():
        start = max(0, event["support_start_seconds"] - config.clip_before)
        end = min(duration, event["last_observed_seconds"] + config.clip_after)
        try:
            event["clip"] = make_clip(
                exports.directory / "annotated.mp4",
                directory / f"event_{event['event_id']:06d}.mp4",
                start,
                end,
                video_info["width"],
                video_info["height"],
                video_info["nominal_fps"],
            )
            event["clip"]["context_truncated"] = (
                event["support_start_seconds"] < config.clip_before
                or event["last_observed_seconds"] + config.clip_after > duration
            )
        except Exception as exc:
            event["quality_flags"].append("clip_failed")
            event["clip_error"] = str(exc)
        exports.update_event(event)
