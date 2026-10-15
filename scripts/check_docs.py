#!/usr/bin/env python3
"""Lightweight documentation checks (run in CI and before release).

Checks, in order:

1. relative Markdown links / HTML ``src``/``href`` attributes resolve on disk
   (fenced code blocks are skipped);
2. repository paths written in backticks (``configs/...``, ``scripts/...``,
   ``docs/...``, ...) exist;
3. ``vlm-dflash <subcommand>`` examples in fenced code blocks only use flags
   the real argparse parser accepts;
4. generated docs (``SUPPORT_MATRIX.md``, ``docs/TRAINING_SIGNALS.md``) match
   their renderers (``--check`` semantics of the render scripts).

Usage: ``python scripts/check_docs.py [--include-internal]``. Internal
research ledgers (``research/``, ``docs/plan/``) are skipped by default; pass
``--include-internal`` to scan them too.
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP_DIRS = {".git", ".venv", "__pycache__", "runs", "assets", ".pytest_cache", "dist", "build"}
INTERNAL_DIRS = {"research", "plan"}  # plan == docs/plan

LINK_RE = re.compile(r"\[[^\]]*\]\(\s*(<[^>]*>|\S+?)(?:\s+[\"'][^\"']*[\"'])?\s*\)")
HTML_RE = re.compile(r'(?:src|href)\s*=\s*"([^"]+)"')
CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
PATH_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".json", ".jsonl", ".sh", ".txt", ".cff", ".svg", ".png"}
KNOWN_PREFIXES = ("configs/", "scripts/", "examples/", "benchmarks/", "tests/", "vlm_dflash/", "docs/", ".github/")
ROOT_DOC_RE = re.compile(r"^[A-Z][A-Z0-9_]*\.(md|json)$")
# Files written by our own tools at run/conversion time (never committed); the
# docs are allowed to reference them as outputs.
GENERATED_ARTIFACTS = {"VLLM_DRAFT.md", "VLLM_BRIDGE.json"}
# Lines that explicitly frame a path as belonging to an upstream/reference
# project are skipped: those files live in the referenced repository, not here.
UPSTREAM_MARKERS = ("upstream", "reference repo", "reference source", "nemo_automodel", "read-only reference")


def iter_markdown(include_internal: bool):
    for path in sorted(ROOT.rglob("*.md")):
        parts = set(path.relative_to(ROOT).parts[:-1])
        if parts & SKIP_DIRS:
            continue
        if not include_internal:
            rel = path.relative_to(ROOT)
            if rel.parts[0] == "research" or (len(rel.parts) > 1 and rel.parts[0] == "docs" and rel.parts[1] == "plan"):
                continue
        yield path


def strip_fences(text: str) -> str:
    """Blank out fenced code blocks so link checks skip examples."""
    lines = text.splitlines(keepends=True)
    out, in_fence = [], False
    fence = None
    for line in lines:
        stripped = line.lstrip()
        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_fence = True
            fence = stripped[:3]
            out.append("\n")
            continue
        if in_fence and stripped.startswith(fence):
            in_fence = False
            fence = None
            out.append("\n")
            continue
        out.append("\n" if in_fence else line)
    return "".join(out)


def code_blocks(text: str):
    """Yield (start_line, fence_language, block_text)."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fence = stripped[:3]
            lang = stripped[3:].strip()
            start = i + 2
            body = []
            i += 1
            while i < len(lines) and not lines[i].lstrip().startswith(fence):
                body.append(lines[i])
                i += 1
            yield start, lang, "\n".join(body)
        i += 1


def link_targets(path: Path, text: str):
    body = strip_fences(text)
    for match in LINK_RE.finditer(body):
        target = match.group(1).strip()
        if target.startswith("<") and target.endswith(">"):
            target = target[1:-1]
        yield target
    for match in HTML_RE.finditer(body):
        yield match.group(1).strip()


def check_links(errors: list[str], include_internal: bool) -> int:
    checked = 0
    for path in iter_markdown(include_internal):
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        for target in link_targets(path, text):
            if target.startswith(("http://", "https://", "mailto:", "#", "/")):
                continue
            if any(ch in target for ch in "<>*{}"):
                continue
            clean = target.split("#", 1)[0]
            if not clean:
                continue
            clean = clean.replace("%20", " ")
            resolved = (path.parent / clean).resolve()
            checked += 1
            if not resolved.exists():
                errors.append(f"{rel}: link target missing: {target}")
    return checked


def check_paths(errors: list[str], include_internal: bool) -> int:
    checked = 0
    for path in iter_markdown(include_internal):
        rel = path.relative_to(ROOT)
        body = strip_fences(path.read_text(encoding="utf-8"))
        lines = body.splitlines()
        for match in CODE_SPAN_RE.finditer(body):
            token = match.group(1).strip()
            if any(ch in token for ch in "<>${*") or token.startswith(("~", "/", "http", "url")):
                continue
            if token in GENERATED_ARTIFACTS:
                continue
            line_index = body.count("\n", 0, match.start())
            line = lines[line_index].lower() if line_index < len(lines) else ""
            previous = lines[line_index - 1].lower() if line_index > 0 else ""
            if any(marker in line or marker in previous for marker in UPSTREAM_MARKERS):
                continue
            candidate = token.split(":", 1)[0]  # allow file.py:12 style
            if Path(candidate).suffix not in PATH_SUFFIXES:
                continue
            is_repo_path = candidate.startswith(KNOWN_PREFIXES) or ROOT_DOC_RE.match(candidate)
            if not is_repo_path:
                continue
            checked += 1
            if not (ROOT / candidate).exists():
                errors.append(f"{rel}: referenced path missing: {candidate}")
    return checked


def _subcommand_options():
    from vlm_dflash.cli import build_parser

    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    options = {}
    for name, child in sub.choices.items():
        flags = {"-h", "--help"}
        for action in child._actions:
            flags.update(action.option_strings)
        options[name] = flags
    options["train"] = options["train"] | {"--feature-mode"}
    return options


FLAG_RE = re.compile(r"--[a-z][a-z0-9-]*")


def check_cli_examples(errors: list[str], include_internal: bool) -> int:
    options = _subcommand_options()
    checked = 0
    for path in iter_markdown(include_internal):
        rel = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        for start, lang, block in code_blocks(text):
            if lang not in {"bash", "sh", "console", "shell", ""}:
                continue
            lines = block.splitlines()
            index = 0
            while index < len(lines):
                line = lines[index]
                if "vlm-dflash" not in line:
                    index += 1
                    continue
                # join line continuations and shell prefixes after &&
                joined = line
                j = index
                while joined.rstrip().endswith("\\") and j + 1 < len(lines):
                    j += 1
                    joined = joined.rstrip()[:-1] + " " + lines[j]
                index = j + 1
                for segment in joined.split("&&"):
                    segment = segment.strip()
                    if not segment.startswith("vlm-dflash") and "vlm-dflash" not in segment.split()[0:2]:
                        continue
                    if "vlm-dflash" != segment.split()[0]:
                        continue
                    parts = segment.split()
                    if len(parts) < 2:
                        continue
                    command = parts[1]
                    if command not in options:
                        errors.append(f"{rel}:{start + j}: unknown vlm-dflash subcommand {command!r}")
                        continue
                    for flag in FLAG_RE.findall(segment):
                        checked += 1
                        if flag not in options[command]:
                            errors.append(f"{rel}:{start + j}: vlm-dflash {command}: unknown flag {flag}")
    return checked


def check_generated(errors: list[str]):
    for module_name, target in (
        ("render_support_matrix.py", "SUPPORT_MATRIX.md"),
        ("render_training_signals.py", "docs/TRAINING_SIGNALS.md"),
    ):
        spec = importlib.util.spec_from_file_location(module_name.replace(".py", ""), ROOT / "scripts" / module_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rendered = module.render()
        current = (ROOT / target).read_text(encoding="utf-8")
        if rendered != current:
            errors.append(f"{target}: generated file is stale (re-run scripts/{module_name})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-internal", action="store_true", help="also scan research/ and docs/plan/")
    args = parser.parse_args()

    errors: list[str] = []
    links = check_links(errors, args.include_internal)
    paths = check_paths(errors, args.include_internal)
    cli = check_cli_examples(errors, args.include_internal)
    check_generated(errors)

    print(f"checked {links} relative links, {paths} referenced paths, {cli} CLI flags")
    if errors:
        print(f"\n{len(errors)} problem(s):", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    print("docs OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
