"""Serve the `web/` directory on the port given by $PORT (or 5175 as a fallback).

Wraps `python -m http.server` so the preview system can assign a free port via
the PORT env var. The plain `http.server` module hard-codes its own argument
list and ignores $PORT, which means a launch.json with `autoPort: true` would
silently bind to whatever port the operator wrote into runtimeArgs, defeating
the point. This script is the smallest thing that still works.
"""

from __future__ import annotations

import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


def main() -> int:
    port = int(os.environ.get("PORT", "5175"))
    directory = sys.argv[1] if len(sys.argv) > 1 else "."
    os.chdir(directory)
    server = HTTPServer(("0.0.0.0", port), SimpleHTTPRequestHandler)
    print(f"Serving {os.getcwd()} on http://0.0.0.0:{port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
