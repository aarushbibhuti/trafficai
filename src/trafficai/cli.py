"""TrafficAI command line; subcommands are usable without editing source files."""

import argparse
import json
import sys
from pathlib import Path

import cv2

from .calibration import Calibration, draw_calibration, fingerprint, select_points
from .config import Config
from .pipeline import analyze, install_signal_handlers
from .video import VideoReader


def parser():
    root = argparse.ArgumentParser(description="TrafficAI: reviewable vehicle motion evidence")
    sub = root.add_subparsers(dest="command", required=True)
    run = sub.add_parser("analyze", help="Track a video and export evidence (headless by default)")
    run.add_argument("video", type=Path)
    run.add_argument("--output", type=Path, default=Path("runs"))
    run.add_argument("--calibration", type=Path)
    run.add_argument("--features", type=Path)
    run.add_argument("--allow-calibration-reuse", action="store_true")
    for name in ("model", "device", "tracker", "units"):
        run.add_argument(
            f"--{name}",
            default=getattr(Config(), name),
            **({"choices": ["kmh", "mph", "mps"]} if name == "units" else {}),
        )
    for name in (
        "confidence",
        "iou",
        "motion_window",
        "min_span",
        "acceleration_window",
        "max_gap",
        "track_ttl",
        "max_speed",
        "max_residual",
        "jump_tolerance",
        "braking_threshold",
        "acceleration_threshold",
        "event_duration",
        "release_ratio",
        "clip_before",
        "clip_after",
        "time_bin",
    ):
        run.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(Config(), name))
    run.add_argument("--grid-size", type=int, default=Config().grid_size)
    run.add_argument("--classes", type=int, nargs="+", default=list(Config().classes))
    run.add_argument("--max-frames", type=int)
    run.add_argument("--preview", action="store_true")
    run.add_argument("--no-clips", action="store_true")

    frame = sub.add_parser("frame", help="Export a source frame for headless calibration")
    frame.add_argument("video", type=Path)
    frame.add_argument("--at", type=float, default=0, help="Elapsed video seconds")
    frame.add_argument("--output", type=Path, required=True)
    frame.add_argument("--overwrite", action="store_true")

    cal = sub.add_parser("calibrate", help="Create/correct a measured road rectangle")
    cal.add_argument("video", type=Path)
    cal.add_argument("--output", type=Path, required=True)
    cal.add_argument("--at", type=float, default=0)
    cal.add_argument("--points", help='Ordered corners: "x1,y1 x2,y2 x3,y3 x4,y4"')
    cal.add_argument("--width-m", type=float)
    cal.add_argument("--length-m", type=float)
    cal.add_argument("--dimensions-source", help="Measurement method or explicit assumption")
    cal.add_argument("--assumptions", help="Record geometric assumptions or known limitations")
    cal.add_argument("--load", type=Path, help="Load calibration to validate or correct")
    cal.add_argument("--interactive", action="store_true", help="Click corners; U undo, R reset")
    cal.add_argument("--allow-calibration-reuse", action="store_true")
    cal.add_argument("--overwrite", action="store_true")
    return root


def get_frame(path, at):
    import math

    if not math.isfinite(at) or at < 0:
        raise ValueError("--at must be a finite nonnegative number of seconds")
    with VideoReader(path) as reader:
        for _, timestamp, frame in reader.frames():
            if timestamp >= at:
                return frame, (reader.width, reader.height)
    raise ValueError(f"No frame at or after {at} seconds")


def save_image(path, frame, overwrite=False):
    # Exclusive creation protects earlier artifacts; encode before opening the destination.
    success, encoded = cv2.imencode(Path(path).suffix or ".png", frame)
    if not success:
        raise ValueError(f"Cannot encode image {path}")
    with open(path, "wb" if overwrite else "xb") as handle:
        handle.write(encoded.tobytes())


def calibrate(args):
    frame, size = get_frame(args.video, args.at)
    source_hash = fingerprint(args.video)
    existing = Calibration.load(args.load) if args.load else None
    if existing:
        existing.check_compatibility(size, source_hash, args.allow_calibration_reuse)
    points = existing.source_points if existing else None
    if args.points:
        points = [[float(x) for x in pair.split(",")] for pair in args.points.split()]
    if args.interactive:
        points = select_points(frame, points)
    width = args.width_m if args.width_m is not None else (existing.width_m if existing else None)
    length = (
        args.length_m if args.length_m is not None else (existing.length_m if existing else None)
    )
    dimensions_source = args.dimensions_source or (existing.dimensions_source if existing else None)
    if points is None or width is None or length is None or dimensions_source is None:
        raise ValueError(
            "Provide four --points (or --interactive), --width-m, --length-m and "
            "--dimensions-source; alternatively --load an existing calibration"
        )
    calibration = Calibration(
        list(size),
        points,
        width,
        length,
        source_hash,
        dimensions_source,
        args.assumptions
        or (
            existing.assumptions
            if existing
            else "Fixed camera; planar road; negligible lens distortion"
        ),
    )
    preview = args.output.with_suffix(".preview.png")
    # Preflight all names before changing any existing artifact.
    if not args.overwrite and (args.output.exists() or preview.exists()):
        raise ValueError("Calibration/preview already exists; choose another output or --overwrite")
    calibration.save(args.output, args.overwrite)
    save_image(preview, draw_calibration(frame, calibration), args.overwrite)
    print(
        f"Saved {args.output} and {preview}. Inspect corner order and dimensions before analysis."
    )


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        install_signal_handlers()
        if args.command == "analyze":
            values = vars(args).copy()
            for key in (
                "command",
                "video",
                "output",
                "calibration",
                "features",
                "allow_calibration_reuse",
            ):
                values.pop(key)
            values["clips"] = not values.pop("no_clips")
            values["classes"] = tuple(values["classes"])
            directory = analyze(
                args.video,
                args.output,
                Config(**values),
                args.calibration,
                args.features,
                args.allow_calibration_reuse,
            )
            print(f"Results: {directory}")
            status = json.loads((directory / "run.json").read_text())["status"]
            return 130 if status == "cancelled" else 0
        if args.command == "frame":
            frame, _ = get_frame(args.video, args.at)
            save_image(args.output, frame, args.overwrite)
            print(args.output)
        elif args.command == "calibrate":
            calibrate(args)
        return 0
    except KeyboardInterrupt:
        print("Cancelled", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"TrafficAI: {exc}", file=sys.stderr)
        return 1
