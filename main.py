"""Compatibility entry point; prefer the installed trafficai command."""

from trafficai.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
