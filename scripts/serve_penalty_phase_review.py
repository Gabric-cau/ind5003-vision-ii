"""Read-only loopback video review server; no uploads or external publishing."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


class Handler(SimpleHTTPRequestHandler):
    def list_directory(self, path):
        self.send_error(403, "Directory listing disabled")
        return None

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def send_head(self):
        self.remaining = None
        path = Path(self.translate_path(self.path)).resolve()
        if not path.is_relative_to(Path(self.directory).resolve()):
            self.send_error(403)
            return None
        if path.suffix.lower() != ".mp4" or not path.is_file():
            return super().send_head()
        length = path.stat().st_size
        start, end = 0, length - 1
        request = self.headers.get("Range")
        if request:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", request.strip())
            if not match or not any(match.groups()):
                self.send_error(416, "Invalid range")
                return None
            if not match[1]:
                start = max(0, length - int(match[2]))
            else:
                start = int(match[1])
                end = min(end, int(match[2])) if match[2] else end
            if start > end or start >= length:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{length}")
                self.end_headers()
                return None
        handle = path.open("rb")
        handle.seek(start)
        self.remaining = end - start + 1
        self.send_response(206 if request else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(self.remaining))
        if request:
            self.send_header("Content-Range", f"bytes {start}-{end}/{length}")
        self.end_headers()
        return handle

    def copyfile(self, source, outputfile):
        try:
            if self.remaining is None:
                return super().copyfile(source, outputfile)
            remaining = self.remaining
            while remaining > 0:
                block = source.read(min(262144, remaining))
                if not block:
                    break
                outputfile.write(block)
                remaining -= len(block)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "outputs/phase_optimization_2026-10-09/review")
    parser.add_argument("--port", type=int, default=8882)
    args = parser.parse_args()
    if not (args.directory / "index.html").is_file():
        parser.error("Build the review package first")
    handler = partial(Handler, directory=str(args.directory.resolve()))
    print(f"Review: http://127.0.0.1:{args.port}/index.html", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
