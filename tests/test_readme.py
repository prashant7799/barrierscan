"""The README must tell the truth: its code runs and its outputs match."""

from __future__ import annotations

import contextlib
import io
import re
import subprocess
import sys
from pathlib import Path

import pytest
from synthetic import random_walk_ohlc

from barrierscan import brute_force_one, first_touch

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text(encoding="utf-8")


def section(heading_start: str) -> str:
    """Lines after the heading that starts with `heading_start`, up to the next
    heading of the same level or higher. Lines inside code fences are never
    headings (a "# comment" in Python code is not a Markdown heading)."""
    level = len(heading_start.split(" ")[0])
    lines = README.splitlines()
    start = next(k for k, line in enumerate(lines) if line.startswith(heading_start))
    in_code = False
    collected = []
    for line in lines[start + 1 :]:
        if line.startswith("```"):
            in_code = not in_code
        elif not in_code and line.startswith("#"):
            this_level = len(line) - len(line.lstrip("#"))
            if this_level <= level and line[this_level : this_level + 1] == " ":
                break
        collected.append(line)
    return "\n".join(collected) + "\n"


def code_blocks(text: str, language: str) -> list[str]:
    return re.findall(rf"```{language}\n(.*?)```", text, flags=re.DOTALL)


def same_output(printed: str, shown: str) -> bool:
    def clean(text):
        return [line.rstrip() for line in text.strip("\n").splitlines()]

    return clean(printed) == clean(shown)


def test_quickstart_runs_and_prints_what_the_readme_shows():
    text = section("## Quickstart (60 seconds)")
    code = code_blocks(text, "python")[0]
    shown = code_blocks(text, "text")[0]
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(code, "README quickstart", "exec"), {})
    assert same_output(printed.getvalue(), shown)


def test_quickstart_example_file_matches_the_readme():
    text = section("## Quickstart (60 seconds)")
    shown = code_blocks(text, "text")[0]
    printed = subprocess.run(
        [sys.executable, "examples/quickstart.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    ).stdout
    assert same_output(printed, shown)


def test_oracle_snippet_runs():
    text = section("### brute_force_one")
    code = code_blocks(text, "python")[0]
    open_, high, low, close = random_walk_ohlc(500, seed=1)
    namespace = {
        "open_": open_,
        "high": high,
        "low": low,
        "close": close,
        "first_touch": first_touch,
        "brute_force_one": brute_force_one,
    }
    exec(compile(code, "README oracle snippet", "exec"), namespace)


@pytest.mark.parametrize("script", ["vol_barriers_floor", "meta_labeling", "purged_cv"])
def test_example_output_matches_the_readme(script):
    link = f"(examples/{script}.py)"
    after_link = README[README.index(link) :]
    shown = code_blocks(after_link, "text")[0]
    printed = subprocess.run(
        [sys.executable, f"examples/{script}.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    ).stdout
    assert same_output(printed, shown), printed


def test_benchmark_example_runs():
    printed = subprocess.run(
        [sys.executable, "examples/benchmark.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    ).stdout
    assert "events per second" in printed
