"""Single publisher: mirror one run to ``<repo>@main:<path>/runs/<run_id>/`` on Hugging Face.

Each cycle uploads files whose content hash changed since the last confirmed upload, in one commit
pinned to the branch head it read (``parent_commit``); a moved head is re-read and retried with
exponential backoff. Completed raw shards are immutable: a shard whose bytes change after upload is
a hard error, never an overwrite. Only paths under this task's directory are written; ``INDEX.json``
is merged with the remote copy, never replaced wholesale. After each commit the remote sizes and
hashes are read back. Pending uploads persist in ``publish_state.json`` so a restart resumes.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import time
from pathlib import Path

from research.qwen36_a100_seven_eval.settings import read_json, save_json

PUBLISHED = (
    "manifest.json",
    "status.json",
    "summary.json",
    "summary.csv",
    "sessions.json",
    "raw/**/*.jsonl",
    "stages/**/*.json",
)
IMMUTABLE = "raw/**/*.jsonl"


def candidates(run: Path) -> dict[str, Path]:
    out = {}
    for pattern in PUBLISHED:
        for path in run.glob(pattern):
            if (
                path.is_file()
                and not path.name.endswith((".partial", ".tmp"))
                and ".invalid" not in path.name
            ):
                out[path.relative_to(run).as_posix()] = path
    return out


def select(value: dict, keys: str) -> dict:
    return {k: value[k] for k in keys.split() if k in value}


def public_payload(rel: str, path: Path) -> bytes:
    """Publish explicit evidence fields; keep site configuration and diagnostics local."""
    if rel == "summary.csv":
        return path.read_bytes()
    if rel.startswith("raw/"):
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        keys = (
            "id turn topic stage protocol method arm gpu_id text tokens wall_seconds "
            "output_tokens prompt_tokens prompt_token_ids_sha256 finish_reason official_counters"
        )
        return "".join(
            json.dumps(select(row, keys), ensure_ascii=False) + "\n" for row in rows
        ).encode()
    value = read_json(path)
    if rel == "manifest.json":
        value = dict(
            select(
                value, "run_id suite methods created code versions protocol publish"
            ),
            target=select(value["target"], "revision files stop_token_ids"),
            exports={
                m: select(
                    e,
                    "repo revision checkpoint_sha256 config_sha256 model_safetensors_sha256",
                )
                for m, e in value["exports"].items()
            },
            data=select(
                value["data"],
                "formal sessions turns commit source test_jsonl_sha256 full_scope image_sha256 references",
            ),
            engine=select(
                value["engine"],
                "gpu_memory_utilization max_num_batched_tokens gdn_decode_kernel moe_policy",
            ),
        )
    elif rel == "sessions.json":
        value = [select(row, "id topic questions image_sha256") for row in value]
    elif rel.startswith("stages/"):
        value = dict(
            select(
                value,
                "status method arm stage protocol topics completed_topics versions gpu_name",
            ),
            method_metadata=select(
                value.get("method_metadata", {}),
                "speculator model queries draft_causal graph_mode graphs",
            ),
            resources=select(
                value.get("resources", {}),
                "status samples ownership_mode continuous_exclusivity_claim extra_app_samples_outside_timed",
            ),
        )
    elif rel == "status.json":
        value = select(
            value,
            "status suite run_id updated jobs jobs_done jobs_failed raw_shards formal",
        )
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode()


class Publisher:
    def __init__(
        self, run: Path, repo: str, prefix: str, api=None, max_attempts: int = 6
    ):
        self.run, self.repo, self.prefix = Path(run), repo, prefix.strip("/")
        self.run_id = read_json(self.run / "manifest.json")["run_id"]
        if not self.prefix.startswith("eval/") or any(
            x in ("", ".", "..") for x in self.prefix.split("/")
        ):
            raise ValueError("publication prefix must be a directory below eval/")
        if (
            not self.run_id
            or Path(self.run_id).name != self.run_id
            or self.run_id in (".", "..")
        ):
            raise ValueError("invalid run id")
        self.remote_root = f"{self.prefix}/runs/{self.run_id}"
        self.state_path = self.run / "publish_state.json"
        self.state = read_json(self.state_path, {"uploaded": {}, "commits": []})
        destination = {"repo": repo, "prefix": self.prefix}
        if self.state.get("destination", destination) != destination:
            raise ValueError("publisher destination differs from the saved state")
        self.state["destination"] = destination
        self.max_attempts = max_attempts
        if api is None:
            from huggingface_hub import HfApi

            api = HfApi()
        self.api = api

    def changed(self) -> dict[str, tuple[bytes, str]]:
        out = {}
        for rel, path in candidates(self.run).items():
            payload = public_payload(rel, path)
            digest = hashlib.sha256(payload).hexdigest()
            old = self.state["uploaded"].get(rel)
            if old == digest:
                continue
            if old is not None and fnmatch.fnmatch(rel, IMMUTABLE):
                raise ValueError(f"immutable raw shard changed after upload: {rel}")
            out[rel] = (payload, digest)
        return out

    def index_operation(self, head: str):
        """INDEX.json merged with the remote copy: this run's entry is added or refreshed."""
        from huggingface_hub import CommitOperationAdd
        from huggingface_hub.errors import EntryNotFoundError

        index = {"runs": {}}
        try:
            local = self.api.hf_hub_download(
                self.repo, f"{self.prefix}/INDEX.json", revision=head
            )
            index = json.loads(Path(local).read_text())
        except EntryNotFoundError:
            pass
        status = read_json(self.run / "status.json", {})
        index.setdefault("runs", {})[self.run_id] = {
            "path": f"runs/{self.run_id}",
            "status": status.get("status"),
            "updated": time.strftime("%F %T"),
            "suite": status.get("suite"),
        }
        payload = json.dumps(index, indent=2, sort_keys=True).encode()
        return CommitOperationAdd(
            path_in_repo=f"{self.prefix}/INDEX.json", path_or_fileobj=payload
        )

    def cycle(self) -> dict:
        from huggingface_hub import CommitOperationAdd
        from huggingface_hub.errors import EntryNotFoundError

        todo = self.changed()
        if not todo:
            return {"uploaded": 0}
        delay = 5
        for attempt in range(1, self.max_attempts + 1):
            try:
                head = self.api.repo_info(self.repo, revision="main").sha
                for rel, (payload, _) in todo.items():
                    if not fnmatch.fnmatch(rel, IMMUTABLE):
                        continue
                    try:
                        existing = self.api.hf_hub_download(
                            self.repo, f"{self.remote_root}/{rel}", revision=head
                        )
                    except EntryNotFoundError:
                        continue
                    if Path(existing).read_bytes() != payload:
                        raise ValueError(f"remote immutable raw shard differs: {rel}")
                ops = [
                    CommitOperationAdd(
                        path_in_repo=f"{self.remote_root}/{rel}",
                        path_or_fileobj=payload,
                    )
                    for rel, (payload, _) in sorted(todo.items())
                ]
                ops.append(self.index_operation(head))
                for op in ops:
                    if not op.path_in_repo.startswith(self.prefix + "/"):
                        raise ValueError(
                            f"refusing to write outside {self.prefix}: {op.path_in_repo}"
                        )
                info = self.api.create_commit(
                    self.repo,
                    operations=ops,
                    parent_commit=head,
                    commit_message=f"eval {self.run_id}: {len(todo)} file(s)",
                )
                self.verify(info.oid, todo)
                for rel, (_, digest) in todo.items():
                    self.state["uploaded"][rel] = digest
                self.state["commits"].append(
                    {"oid": info.oid, "files": len(todo), "at": time.strftime("%F %T")}
                )
                save_json(self.state_path, self.state)
                return {"uploaded": len(todo), "commit": info.oid, "attempts": attempt}
            except Exception as error:
                self.state["last_error"] = f"{type(error).__name__}: {error}"[:500]
                save_json(self.state_path, self.state)
                if attempt == self.max_attempts:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 300)
        return {"uploaded": 0}

    def verify(self, revision: str, todo: dict) -> None:
        """Verify the bytes at the returned commit, including files stored as ordinary Git blobs."""
        for rel, (payload, digest) in todo.items():
            path = self.api.hf_hub_download(
                self.repo, f"{self.remote_root}/{rel}", revision=revision
            )
            content = Path(path).read_bytes()
            if (
                len(content) != len(payload)
                or hashlib.sha256(content).hexdigest() != digest
            ):
                raise ValueError(f"remote content of {rel} differs")


def check_write_access(repo: str, api=None) -> dict:
    """Preflight: authenticated user and a dry write probe is not possible on HF; check role instead."""
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    who = api.whoami()
    info = api.repo_info(repo)
    owner = repo.split("/")[0]
    orgs = {o.get("name") for o in who.get("orgs", [])}
    allowed = who.get("name") == owner or owner in orgs
    token_role = (who.get("auth") or {}).get("accessToken", {}).get("role")
    return {
        "user": who.get("name"),
        "repo": repo,
        "head": info.sha,
        "owner_or_org_member": allowed,
        "token_role": token_role,
        "ok": bool(allowed and token_role in (None, "write", "fineGrained")),
    }


def loop(
    run: Path, repo: str, prefix: str, interval: int, until_done: bool = True
) -> int:
    publisher = Publisher(run, repo, prefix)
    while True:
        result = publisher.cycle()
        print(time.strftime("%F %T"), "publish", json.dumps(result), flush=True)
        status = read_json(Path(run) / "status.json", {}).get("status")
        if (
            until_done
            and status in ("complete", "partial", "failed", "blocked")
            and not publisher.changed()
        ):
            return 0
        time.sleep(interval)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--path",
        required=True,
        help="task directory inside the repo, e.g. eval/qwen36_a100_seven_v1",
    )
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    if args.once:
        print(json.dumps(Publisher(args.run_dir, args.repo, args.path).cycle()))
        return 0
    return loop(args.run_dir, args.repo, args.path, args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
