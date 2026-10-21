"""Minimal OpenAI-compatible HTTP client used by the lifecycle harness.

No third-party dependency is required.  The client deliberately speaks the
public ``/v1/chat/completions`` protocol and opts into vLLM's
``return_token_ids`` extension so token-level correctness can be checked from
the same HTTP path that partners use.
"""

from __future__ import annotations

import http.client
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

SUCCESS_FINISH_REASONS = {"stop", "length"}


class ServingHTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


class StreamProtocolError(RuntimeError):
    """An SSE stream violated the supported OpenAI-compatible protocol."""

    def __init__(self, message: str, kind: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class StreamEvent:
    index: int
    t_s: float
    content: str = ""
    token_ids: list[int] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    raw: Optional[dict[str, Any]] = None
    event_type: str = "delta"

    @property
    def has_content(self) -> bool:
        return bool(self.content) or bool(self.token_ids)


@dataclass
class ChatResult:
    request_id: str
    mode: str
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    started_at: float = 0.0
    first_content_s: Optional[float] = None
    last_content_s: Optional[float] = None
    e2e_s: Optional[float] = None
    chunk_gaps_s: list[float] = field(default_factory=list)
    chunk_token_counts: list[Optional[int]] = field(default_factory=list)
    content_events: list[tuple[float, str, list[int]]] = field(default_factory=list)
    num_content_chunks: int = 0
    cancelled: bool = False
    cancel_requested_s: Optional[float] = None
    error: Optional[str] = None
    http_status: Optional[int] = None
    response_id: Optional[str] = None
    stream_state: str = "not_stream"
    done_seen: bool = False
    finish_seen: bool = False
    terminal_status: str = "incomplete"
    truncated: bool = False
    protocol_error: Optional[str] = None
    malformed_events: int = 0
    sse_error_events: int = 0
    token_ids_complete: bool = True
    content_chunks_without_token_ids: int = 0

    def is_successful(self) -> bool:
        return bool(
            self.error is None
            and not self.cancelled
            and self.terminal_status == "completed"
            and self.finish_reason in SUCCESS_FINISH_REASONS
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "mode": self.mode,
            "text": self.text,
            "token_ids": self.token_ids,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "first_content_s": self.first_content_s,
            "last_content_s": self.last_content_s,
            "e2e_s": self.e2e_s,
            "chunk_gaps_s": self.chunk_gaps_s,
            "chunk_token_counts": self.chunk_token_counts,
            "num_content_chunks": self.num_content_chunks,
            "cancelled": self.cancelled,
            "cancel_requested_s": self.cancel_requested_s,
            "error": self.error,
            "http_status": self.http_status,
            "response_id": self.response_id,
            "stream_state": self.stream_state,
            "done_seen": self.done_seen,
            "finish_seen": self.finish_seen,
            "terminal_status": self.terminal_status,
            "truncated": self.truncated,
            "protocol_error": self.protocol_error,
            "malformed_events": self.malformed_events,
            "sse_error_events": self.sse_error_events,
            "token_ids_complete": self.token_ids_complete,
            "content_chunks_without_token_ids": self.content_chunks_without_token_ids,
        }


class ServingClient:
    def __init__(self, base_url: str, timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    # ------------------------------------------------------------------
    # Simple endpoints
    # ------------------------------------------------------------------

    def _open(self, path: str, *, body: Optional[dict[str, Any]] = None, timeout_s: Optional[float] = None):
        url = self.base_url + path
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
        return urllib.request.urlopen(request, timeout=timeout_s or self.timeout_s)

    def health(self) -> bool:
        try:
            with self._open("/health", timeout_s=3.0) as response:
                return response.status == 200
        except Exception:
            return False

    def models(self) -> dict[str, Any]:
        with self._open("/v1/models", timeout_s=5.0) as response:
            return json.loads(response.read().decode("utf-8"))

    def metrics_text(self) -> str:
        with self._open("/metrics", timeout_s=5.0) as response:
            return response.read().decode("utf-8")

    def wait_for_health(self, timeout_s: float = 300.0, poll_s: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.health():
                return True
            time.sleep(poll_s)
        return False

    # ------------------------------------------------------------------
    # Chat completions
    # ------------------------------------------------------------------

    def _chat_result_from_json(self, body: dict[str, Any], request_id: str, mode: str, started: float) -> ChatResult:
        if body.get("error"):
            return ChatResult(
                request_id=request_id,
                mode=mode,
                started_at=started,
                e2e_s=time.monotonic() - started,
                error=f"response error: {body['error']}",
                terminal_status="error",
                protocol_error="response_error",
                response_id=body.get("id"),
            )
        if not body.get("choices"):
            return ChatResult(
                request_id=request_id,
                mode=mode,
                started_at=started,
                e2e_s=time.monotonic() - started,
                error="response has no choices",
                terminal_status="error",
                protocol_error="missing_choices",
                response_id=body.get("id"),
            )
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        token_ids = choice.get("token_ids")
        if token_ids is None:
            token_ids = message.get("token_ids")
        finish_reason = choice.get("finish_reason")
        result = ChatResult(
            request_id=request_id,
            mode=mode,
            text=content,
            token_ids=list(token_ids or []),
            finish_reason=finish_reason,
            usage=body.get("usage"),
            started_at=started,
            first_content_s=time.monotonic() - started if content or token_ids else None,
            last_content_s=time.monotonic() - started if content or token_ids else None,
            e2e_s=time.monotonic() - started,
            response_id=body.get("id"),
            done_seen=True,
            finish_seen=finish_reason is not None,
            token_ids_complete=token_ids is not None,
        )
        if finish_reason not in SUCCESS_FINISH_REASONS:
            result.error = f"non-success finish_reason {finish_reason!r}"
            result.protocol_error = "invalid_finish_reason"
            result.terminal_status = "error"
        else:
            result.terminal_status = "completed"
        return result

    def chat(self, body: dict[str, Any], *, mode: str = "unknown", request_id: Optional[str] = None) -> ChatResult:
        request_id = request_id or str(body.get("request_id") or "request")
        started = time.monotonic()
        try:
            with self._open("/v1/chat/completions", body=body) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return self._chat_result_from_json(payload, request_id, mode, started)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return ChatResult(
                request_id=request_id,
                mode=mode,
                started_at=started,
                e2e_s=time.monotonic() - started,
                error=f"HTTP {exc.code}: {detail[:1000]}",
                http_status=exc.code,
                terminal_status="error",
            )
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            return ChatResult(
                request_id=request_id,
                mode=mode,
                started_at=started,
                e2e_s=time.monotonic() - started,
                error=f"transport: {exc}",
                terminal_status="error",
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            return ChatResult(
                request_id=request_id,
                mode=mode,
                started_at=started,
                e2e_s=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
                terminal_status="error",
            )

    def iter_stream(self, body: dict[str, Any], *, mode: str = "unknown", request_id: Optional[str] = None) -> Iterable[StreamEvent]:
        request_id = request_id or str(body.get("request_id") or "request")
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        started = time.monotonic()
        response = urllib.request.urlopen(request, timeout=self.timeout_s)
        done_seen = False
        finish_seen = False
        try:
            index = 0
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done_seen = True
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise StreamProtocolError(
                        f"malformed SSE JSON at event {index}: {exc}",
                        "malformed_json",
                    ) from exc
                if payload.get("error"):
                    raise StreamProtocolError(f"SSE error event: {payload['error']}", "sse_error")
                choices = payload.get("choices") or []
                if not choices:
                    if payload.get("usage") is not None:
                        yield StreamEvent(
                            index=index,
                            t_s=time.monotonic() - started,
                            usage=payload.get("usage"),
                            raw=payload,
                            event_type="usage",
                        )
                        index += 1
                        continue
                    raise StreamProtocolError("SSE event has no choices and no usage", "malformed_event")
                choice = choices[0]
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                token_ids = choice.get("token_ids")
                if token_ids is None:
                    token_ids = delta.get("token_ids") or []
                finish_reason = choice.get("finish_reason")
                if finish_reason is not None:
                    if finish_seen:
                        raise StreamProtocolError("SSE stream has multiple finish_reason events", "duplicate_finish")
                    finish_seen = True
                event = StreamEvent(
                    index=index,
                    t_s=time.monotonic() - started,
                    content=content,
                    token_ids=list(token_ids),
                    finish_reason=finish_reason,
                    usage=payload.get("usage"),
                    raw=payload,
                )
                index += 1
                yield event
            if not done_seen:
                raise StreamProtocolError("SSE stream ended before [DONE]", "truncated")
            if not finish_seen:
                raise StreamProtocolError("SSE stream completed without finish_reason", "missing_finish_reason")
        finally:
            response.close()

    def collect_stream(
        self,
        body: dict[str, Any],
        *,
        mode: str = "unknown",
        request_id: Optional[str] = None,
        cancel_after_first_content: bool = False,
    ) -> ChatResult:
        request_id = request_id or str(body.get("request_id") or "request")
        result = ChatResult(request_id=request_id, mode=mode, started_at=time.monotonic(), stream_state="stream")
        generator = self.iter_stream(body, mode=mode, request_id=request_id)
        try:
            for event in generator:
                if event.event_type == "usage":
                    result.usage = event.usage
                    if event.raw and event.raw.get("id"):
                        result.response_id = event.raw["id"]
                    continue
                if event.usage:
                    result.usage = event.usage
                if event.finish_reason:
                    result.finish_reason = event.finish_reason
                    result.finish_seen = True
                if event.raw and event.raw.get("id"):
                    result.response_id = event.raw["id"]
                if not event.has_content:
                    continue
                if result.first_content_s is None:
                    result.first_content_s = event.t_s
                else:
                    result.chunk_gaps_s.append(max(0.0, event.t_s - (result.last_content_s or event.t_s)))
                result.last_content_s = event.t_s
                result.num_content_chunks += 1
                result.content_events.append((event.t_s, event.content, list(event.token_ids)))
                result.text += event.content
                result.token_ids.extend(event.token_ids)
                if event.token_ids:
                    result.chunk_token_counts.append(len(event.token_ids))
                else:
                    result.chunk_token_counts.append(None)
                    result.token_ids_complete = False
                    result.content_chunks_without_token_ids += 1
                if cancel_after_first_content:
                    result.cancelled = True
                    result.cancel_requested_s = time.monotonic() - result.started_at
                    result.terminal_status = "cancelled"
                    break
            else:
                result.done_seen = True
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            result.error = f"HTTP {exc.code}: {detail[:1000]}"
            result.http_status = exc.code
            result.terminal_status = "error"
        except StreamProtocolError as exc:
            result.error = f"protocol {exc.kind}: {exc}"
            result.protocol_error = exc.kind
            result.truncated = exc.kind == "truncated"
            if exc.kind == "malformed_json":
                result.malformed_events += 1
            if exc.kind == "sse_error":
                result.sse_error_events += 1
            result.terminal_status = "truncated" if result.truncated else "error"
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            result.error = f"transport: {exc}"
            result.terminal_status = "error"
        except http.client.IncompleteRead as exc:
            result.error = f"transport truncated: {exc}"
            result.truncated = True
            result.protocol_error = "truncated"
            result.terminal_status = "truncated"
        except Exception as exc:  # pragma: no cover - defensive boundary
            result.error = f"{type(exc).__name__}: {exc}"
            result.terminal_status = "error"
        finally:
            generator.close()
            result.e2e_s = time.monotonic() - result.started_at
        if result.cancelled:
            result.terminal_status = "cancelled"
        elif result.error is None:
            if not result.done_seen:
                result.error = "protocol truncated: stream ended before [DONE]"
                result.protocol_error = "truncated"
                result.truncated = True
                result.terminal_status = "truncated"
            elif not result.finish_seen:
                result.error = "protocol missing_finish_reason: stream ended without finish_reason"
                result.protocol_error = "missing_finish_reason"
                result.terminal_status = "error"
            elif result.finish_reason not in SUCCESS_FINISH_REASONS:
                result.error = f"non-success finish_reason {result.finish_reason!r}"
                result.protocol_error = "invalid_finish_reason"
                result.terminal_status = "error"
            else:
                result.terminal_status = "completed"
        return result


def result_timing_fields(result: ChatResult) -> dict[str, Any]:
    """Derive contract-compliant latency fields from a client result."""

    token_count: Optional[int] = len(result.token_ids) if result.token_ids else None
    if token_count is None and result.usage:
        reported = result.usage.get("completion_tokens")
        if isinstance(reported, int) and reported >= 0:
            token_count = reported
    tpot = None
    tpot_reason = (
        "SSE exposes chunk timestamps, not per-token timestamps; exact TPOT/ITL is unavailable"
    )
    if result.stream_state != "stream":
        tpot_reason = "non-stream responses expose no per-token timestamps; exact TPOT/ITL is unavailable"
    client_proxy = None
    if (
        result.token_ids
        and result.token_ids_complete
        and result.first_content_s is not None
        and result.last_content_s is not None
        and token_count is not None
        and token_count > 1
    ):
        client_proxy = max(0.0, result.last_content_s - result.first_content_s) / (token_count - 1)
    chunk_gap_mean = None
    if len(result.chunk_gaps_s) > 0:
        chunk_gap_mean = sum(result.chunk_gaps_s) / len(result.chunk_gaps_s)
    return {
        "ttft_s": result.first_content_s,
        "e2e_s": result.e2e_s,
        "tpot_s": tpot,
        "tpot_unavailable_reason": tpot_reason,
        "client_visible_token_tpot_proxy_s": client_proxy,
        "client_visible_token_tpot_proxy_definition": (
            "visible token_ids only: (last_content-first_content)/(visible_tokens-1); "
            "assumes uniform token placement inside chunks and is not exact TPOT"
        ),
        "client_chunk_gap_mean_s": chunk_gap_mean,
        "cancel_requested_s": result.cancel_requested_s,
        "num_content_chunks": result.num_content_chunks,
        "num_output_tokens": token_count,
        "num_visible_token_ids": len(result.token_ids),
        "token_ids_complete": result.token_ids_complete,
        "content_chunks_without_token_ids": result.content_chunks_without_token_ids,
        "chunk_token_counts": result.chunk_token_counts,
        "chunk_gap_count": len(result.chunk_gaps_s),
    }
