#!/usr/bin/env python3
"""Serve an audio2strudel output folder to strudel.cc as samples.

  python serve_samples.py out/<song>

Then, in Strudel:  samples('http://localhost:5432/strudel.json')

Only the standard library. Adds the CORS headers a page on https://strudel.cc
needs to fetch from localhost (the same thing `npx @strudel/sampler` does).
"""

import argparse
import functools
import socket
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def log_message(self, fmt, *args):
        if args and str(args[1]) not in ("200", "204", "304"):
            super().log_message(fmt, *args)


def main() -> None:
    p = argparse.ArgumentParser(description="Serve an audio2strudel folder to Strudel as samples.")
    p.add_argument("folder", type=Path, help="The folder with strudel.json, e.g. out/song")
    p.add_argument("--port", type=int, default=5432)
    args = p.parse_args()
    folder = args.folder.resolve()
    if not (folder / "strudel.json").exists():
        sys.exit(f"There is no strudel.json in {folder}. Run to_strudel.py with --as-audio first.")
    handler = functools.partial(Handler, directory=str(folder))
    # Loopback only, on IPv4 and (when available) IPv6: browsers may resolve
    # "localhost" to either.
    servers = [ThreadingHTTPServer(("127.0.0.1", args.port), handler)]
    try:
        class V6(ThreadingHTTPServer):
            address_family = socket.AF_INET6
        servers.append(V6(("::1", args.port), handler))
    except OSError:
        pass
    for extra in servers[1:]:
        threading.Thread(target=extra.serve_forever, daemon=True).start()
    server = servers[0]
    print(f"Serving samples from {folder}")
    print(f"In Strudel: samples('http://localhost:{args.port}/strudel.json')")
    print("Stop with Ctrl+C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
