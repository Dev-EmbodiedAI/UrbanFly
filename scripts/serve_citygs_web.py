from __future__ import annotations

import argparse
import mimetypes
import urllib.parse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class CityGSHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".json": "application/json",
        ".ksplat": "application/octet-stream",
        ".ply": "application/octet-stream",
    }

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_HEAD(self) -> None:
        if self.path.startswith("/abs/"):
            self.serve_absolute_file(head_only=True)
            return
        super().do_HEAD()

    def do_GET(self) -> None:
        if self.path.startswith("/abs/"):
            self.serve_absolute_file(head_only=False)
            return
        super().do_GET()

    def serve_absolute_file(self, head_only: bool) -> None:
        encoded_path = self.path[len("/abs/") :]
        decoded_path = urllib.parse.unquote(encoded_path)
        target = Path(decoded_path)
        if not target.is_file():
            self.send_error(404, f"File not found: {target}")
            return

        mime_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        file_size = target.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", mime_type)
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        if head_only:
            return
        with target.open("rb") as handle:
            self.copyfile(handle, self.wfile)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve local CityGS browser capture pages.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    handler = partial(CityGSHandler, directory=str(root))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {root} at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
