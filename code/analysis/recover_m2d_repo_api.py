#!/usr/bin/env python3
"""Recover a pinned public GitHub tree when remote Git pack transfer stalls."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from pathlib import Path


OWNER = "Snehitc"
REPO = "AMR-encoder-exploration"
COMMIT = "90a67fcfd0c4afe3f4cd11506d20e4886b7a7133"
TARGET = Path("/private/research-artifact")
API = f"https://api.github.com/repos/{OWNER}/{REPO}"


def get_json(url: str):
    last_error = None
    for attempt in range(1, 9):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AMR-R1-audit"})
            with urllib.request.urlopen(req, timeout=90) as response:
                return json.loads(response.read())
        except Exception as error:  # bounded transport retry for truncated API responses
            last_error = error
            print(f"retry={attempt}/8 url={url.rsplit('/', 1)[-1]} error={type(error).__name__}", flush=True)
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"GitHub API request failed after retries: {url}") from last_error


def fetch_blob(sha: str, destination: Path) -> bytes:
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/git/blobs/{sha}"
    subprocess.run(
        [
            "curl", "-fsSL", "--retry", "5", "--retry-delay", "1", "--retry-all-errors",
            "--max-time", "180", "-H", "Accept: application/vnd.github.raw+json",
            "-o", str(destination), url,
        ],
        check=True,
    )
    return destination.read_bytes()


def main() -> None:
    partial = TARGET.with_name(TARGET.name + ".partial_git")
    if TARGET.exists():
        if any(TARGET.iterdir()):
            alternate = TARGET.with_name(TARGET.name + ".partial_api")
            index = 2
            while alternate.exists():
                alternate = TARGET.with_name(TARGET.name + f".partial_api_{index}")
                index += 1
            TARGET.rename(alternate)
            partial = alternate
        else:
            TARGET.rmdir()
    TARGET.mkdir(parents=True)
    tree = get_json(f"{API}/git/trees/{COMMIT}?recursive=1")["tree"]
    files = [entry for entry in tree if entry["type"] == "blob"]
    for index, entry in enumerate(files, start=1):
        path = TARGET / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = fetch_blob(entry["sha"], path)
        if len(payload) != int(entry["size"]):
            raise RuntimeError(f"size mismatch for {entry['path']}: {len(payload)} != {entry['size']}")
        if index % 10 == 0 or index == len(files):
            print(f"downloaded={index}/{len(files)}", flush=True)
    subprocess.run(["git", "init", "-q"], cwd=TARGET, check=True)
    subprocess.run(["git", "config", "user.name", "AMR R1 audit"], cwd=TARGET, check=True)
    subprocess.run(["git", "config", "user.email", "amr-r1-audit@example.invalid"], cwd=TARGET, check=True)
    subprocess.run(["git", "add", "-A"], cwd=TARGET, check=True)
    subprocess.run(["git", "commit", "-q", "-m", f"Pinned upstream snapshot {COMMIT}"], cwd=TARGET, check=True)
    subprocess.run(["git", "remote", "add", "origin", f"https://github.com/{OWNER}/{REPO}.git"], cwd=TARGET, check=True)
    print(json.dumps({
        "upstream_commit": COMMIT,
        "files": len(files),
        "local_snapshot_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=TARGET, text=True).strip(),
        "partial_path": str(partial),
    }))


if __name__ == "__main__":
    main()
