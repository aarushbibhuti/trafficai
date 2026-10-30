"""Compatibility entry point: python src/run_yolo.py analyze VIDEO [options]."""

from trafficai.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
