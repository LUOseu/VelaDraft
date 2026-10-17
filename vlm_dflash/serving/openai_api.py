"""Minimal OpenAI-compatible HTTP client used by the lifecycle harness.

No third-party dependency is required.  The client deliberately speaks the
public ``/v1/chat/completions`` protocol and opts into vLLM's
``return_token_ids`` extension so token-level correctness can be checked from
the same HTTP path that partners use.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


class ServingHTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


@dataclass
class StreamEvent:
    index: int
    t_s: float
    content: str = ""
    token_ids: list[int] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage: Optional[dict[str, Any]] = None
    raw: Optional[dict[str, Any]] = None

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
    chunk_token_counts: list[int] = field(default_factory=list)
    num_content_chunks: int = 0
    cancelled: bool = False
    cancel_requested_s: Optional[float] = None
    error: Optional[str] = None
    http_status: Optional[int] = None
    response_id: Optional[str] = None

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
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        token_ids = choice.get("token_ids")
        if token_ids is None:
            token_ids = message.get("token_ids")
        return ChatResult(
            request_id=request_id,
            mode=mode,
            text=content,
            token_ids=list(token_ids or []),
            finish_reason=choice.get("finish_reason"),
            usage=body.get("usage"),
            started_at=started,
            first_content_s=time.monotonic() - started if content or token_ids else None,
            last_content_s=time.monotonic() - started if content or token_ids else None,
            e2e_s=time.monotonic() - started,
            response_id=body.get("id"),
        )

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
            )
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            return ChatResult(
                request_id=request_id,
                mode=mode,
                started_at=started,
                e2e_s=time.monotonic() - started,
                error=f"transport: {exc}",
            )
        except Exception as exc:  # pragma: no cover - defensive boundary
            return ChatResult(
                request_id=request_id,
                mode=mode,
                started_at=started,
                e2e_s=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
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
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choice = (payload.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                token_ids = choice.get("token_ids")
                if token_ids is None:
                    token_ids = delta.get("token_ids") or []
                event = StreamEvent(
                    index=index,
                    t_s=time.monotonic() - started,
                    content=content,
                    token_ids=list(token_ids),
                    finish_reason=choice.get("finish_reason"),
                    usage=payload.get("usage"),
                    raw=payload,
                )
                index += 1
                yield event
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
        result = ChatResult(request_id=request_id, mode=mode, started_at=time.monotonic())
        generator = self.iter_stream(body, mode=mode, request_id=request_id)
        try:
            for event in generator:
                if event.usage:
                    result.usage = event.usage
                if event.finish_reason:
                    result.finish_reason = event.finish_reason
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
                result.content_events = getattr(result, "content_events", []) + [(event.t_s, event.content, list(event.token_ids))]
                result.text += event.content
                result.token_ids.extend(event.token_ids)
                result.chunk_token_counts.append(len(event.token_ids) or (1 if event.content else 0))
                if cancel_after_first_content:
                    result.cancelled = True
                    result.cancel_requested_s = time.monotonic() - result.started_at
                    break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            result.error = f"HTTP {exc.code}: {detail[:1000]}"
            result.http_status = exc.code
        except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
            result.error = f"transport: {exc}"
        except Exception as exc:  # pragma: no cover - defensive boundary
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            generator.close()
            result.e2e_s = time.monotonic() - result.started_at
        if result.usage and not result.token_ids:
            completion_tokens = result.usage.get("completion_tokens")
            if isinstance(completion_tokens, int):
                result.chunk_token_counts = [completion_tokens]
        return result


def result_timing_fields(result: ChatResult) -> dict[str, Any]:
    """Derive contract-compliant latency fields from a client result."""

    token_count = len(result.token_ids)
    if token_count == 0 and result.chunk_token_counts:
        token_count = sum(result.chunk_token_counts)
    tpot = None
    if (
        result.first_content_s is not None
        and result.last_content_s is not None
        and token_count > 1
    ):
        tpot = max(0.0, result.last_content_s - result.first_content_s) / (token_count - 1)
    proxy = None
    if len(result.chunk_gaps_s) > 0:
        proxy = sum(result.chunk_gaps_s) / len(result.chunk_gaps_s)
    return {
        "ttft_s": result.first_content_s,
        "e2e_s": result.e2e_s,
        "tpot_s": tpot,
        "client_chunk_tpot_proxy_s": proxy,
        "cancel_requested_s": result.cancel_requested_s,
        "num_content_chunks": result.num_content_chunks,
        "num_output_tokens": token_count,
    }
