from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vlm_dflash.serving.openai_api import ServingClient


def _chunk(content="", token_ids=None, finish=None):
    return {
        "id": "chatcmpl-test",
        "choices": [
            {
                "delta": {"content": content, "token_ids": token_ids or []},
                "finish_reason": finish,
            }
        ],
    }


class _FailureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        return

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        scenario = request.get("scenario")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            if scenario == "partial":
                self._send(_chunk("hello", [1]))
            elif scenario == "sse_error":
                self._send(_chunk("hello", [1]))
                self._send({"error": {"message": "engine exploded"}})
                self._send_done()
            elif scenario == "malformed":
                self._send(_chunk("hello", [1]))
                self.wfile.write(b"data: {not-json\n\n")
                self.wfile.flush()
            elif scenario == "duplicate_finish":
                self._send(_chunk("hello", [1], "stop"))
                self._send(_chunk("", [], "stop"))
                self._send_done()
            elif scenario == "empty_ok":
                self._send(_chunk("", [], "stop"))
                self._send_done()
            elif scenario == "missing_finish":
                self._send_done()
            else:
                self._send(_chunk("hello", [1], "stop"))
                self._send_done()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send(self, payload):
        self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode("utf-8"))
        self.wfile.flush()

    def _send_done(self):
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def _server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FailureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _collect(scenario):
    server, thread = _server()
    try:
        client = ServingClient(f"http://127.0.0.1:{server.server_port}", timeout_s=5)
        return client.collect_stream(
            {"model": "m", "messages": [], "stream": True, "scenario": scenario},
            mode="ar",
            request_id=scenario,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_partial_eof_is_truncated_not_success():
    result = _collect("partial")

    assert result.error
    assert result.truncated is True
    assert result.protocol_error == "truncated"
    assert result.terminal_status == "truncated"
    assert result.is_successful() is False


def test_sse_error_event_is_protocol_error():
    result = _collect("sse_error")

    assert "engine exploded" in (result.error or "")
    assert result.protocol_error == "sse_error"
    assert result.is_successful() is False


def test_malformed_json_is_protocol_error():
    result = _collect("malformed")

    assert result.protocol_error == "malformed_json"
    assert result.malformed_events == 1
    assert result.is_successful() is False


def test_duplicate_finish_is_protocol_error():
    result = _collect("duplicate_finish")

    assert result.protocol_error == "duplicate_finish"
    assert result.is_successful() is False


def test_empty_output_with_finish_and_done_is_success():
    result = _collect("empty_ok")

    assert result.error is None
    assert result.text == ""
    assert result.token_ids == []
    assert result.finish_reason == "stop"
    assert result.done_seen is True
    assert result.terminal_status == "completed"
    assert result.is_successful() is True


def test_done_without_finish_is_not_success():
    result = _collect("missing_finish")

    assert result.protocol_error == "missing_finish_reason"
    assert result.is_successful() is False
