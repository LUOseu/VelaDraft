from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vlm_dflash.serving.openai_api import ServingClient


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        return

    def do_GET(self):
        if self.path == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        if request.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            chunks = [
                {"id": "chatcmpl-test", "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
                {"id": "chatcmpl-test", "choices": [{"delta": {"content": "hello", "token_ids": [1]}, "finish_reason": None}]},
                {"id": "chatcmpl-test", "choices": [{"delta": {"content": " world", "token_ids": [2]}, "finish_reason": "stop"}]},
            ]
            try:
                for chunk in chunks:
                    self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(0.01)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        payload = {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello world", "token_ids": [1, 2]},
                    "finish_reason": "stop",
                    "token_ids": [1, 2],
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_stream_parser_and_token_ids():
    server, thread = _server()
    try:
        client = ServingClient(f"http://127.0.0.1:{server.server_port}", timeout_s=5)
        result = client.collect_stream({"model": "m", "messages": [], "stream": True}, mode="ar", request_id="r1")
        assert result.error is None
        assert result.text == "hello world"
        assert result.token_ids == [1, 2]
        assert result.finish_reason == "stop"
        assert result.first_content_s is not None
        assert result.e2e_s is not None
        assert result.num_content_chunks == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_stream_cancel_after_first_content():
    server, thread = _server()
    try:
        client = ServingClient(f"http://127.0.0.1:{server.server_port}", timeout_s=5)
        result = client.collect_stream(
            {"model": "m", "messages": [], "stream": True},
            mode="ar",
            request_id="r2",
            cancel_after_first_content=True,
        )
        assert result.cancelled is True
        assert result.token_ids == [1]
        assert result.cancel_requested_s is not None
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_non_stream_parser():
    server, thread = _server()
    try:
        client = ServingClient(f"http://127.0.0.1:{server.server_port}", timeout_s=5)
        result = client.chat({"model": "m", "messages": []}, mode="ar", request_id="r3")
        assert result.error is None
        assert result.token_ids == [1, 2]
        assert result.text == "hello world"
        assert result.usage["completion_tokens"] == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
