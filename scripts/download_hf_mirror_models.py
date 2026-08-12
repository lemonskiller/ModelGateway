#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


MIRROR = "https://hf-mirror.com"
DEFAULT_ROOT = Path("/nfs/wxz/source/huggingface")
USER_AGENT = "Mozilla/5.0"

REPOS = [
    ("BlivionIaG/Qwen3.5-0.8B-AWQ-INT4", "Qwen3.5-0.8B-AWQ-INT4"),
    ("cyankiwi/Qwen3.5-9B-AWQ-4bit", "Qwen3.5-9B-AWQ-4bit"),
    ("cyankiwi/Qwen3.6-27B-AWQ-INT4", "Qwen3.6-27B-AWQ-INT4"),
    ("sahilchachra/Unlimited-OCR-AWQ", "Unlimited-OCR-AWQ"),
]


def request(url: str, *, method: str = "GET", headers: dict[str, str] | None = None):
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    req = urllib.request.Request(url, method=method, headers=merged)
    return urllib.request.urlopen(req, timeout=60)


def model_info(repo: str) -> dict:
    url = f"{MIRROR}/api/models/{repo}"
    with request(url, headers={"Accept": "application/json"}) as resp:
        return json.load(resp)


def head_size(repo: str, filename: str) -> int | None:
    url = resolve_url(repo, filename)
    try:
        with request(url, method="HEAD") as resp:
            value = resp.headers.get("content-length")
            return int(value) if value else None
    except urllib.error.HTTPError as exc:
        if exc.code == 405:
            return None
        raise


def resolve_url(repo: str, filename: str) -> str:
    quoted = "/".join(urllib.parse.quote(part) for part in filename.split("/"))
    return f"{MIRROR}/{repo}/resolve/main/{quoted}"


def list_files(repo: str) -> list[str]:
    info = model_info(repo)
    files = [s["rfilename"] for s in info.get("siblings", [])]
    return [f for f in files if f and not f.endswith("/")]


def should_skip(path: Path, expected_size: int | None) -> bool:
    if not path.exists():
        return False
    if expected_size is None:
        return path.stat().st_size > 0
    return path.stat().st_size == expected_size


def download_file(repo: str, filename: str, out_path: Path, expected_size: int | None) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = out_path.with_name(out_path.name + ".part")

    if should_skip(out_path, expected_size):
        return f"skip {out_path}"

    resume_from = part_path.stat().st_size if part_path.exists() else 0
    if expected_size is not None and resume_from == expected_size:
        part_path.replace(out_path)
        return f"done {out_path}"
    if expected_size is not None and resume_from > expected_size:
        part_path.unlink()
        resume_from = 0

    headers = {}
    mode = "wb"
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
        mode = "ab"

    url = resolve_url(repo, filename)
    last_log = time.monotonic()
    downloaded = resume_from
    label = f"{repo}/{filename}"

    for attempt in range(1, 6):
        try:
            with request(url, headers=headers) as resp, part_path.open(mode) as fh:
                if resume_from and resp.status != 206:
                    raise RuntimeError(f"server ignored range request for {label}")
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    now = time.monotonic()
                    if now - last_log >= 30:
                        if expected_size:
                            pct = downloaded * 100 / expected_size
                            print(f"{label}: {downloaded / 1024**3:.2f}/{expected_size / 1024**3:.2f} GiB ({pct:.1f}%)", flush=True)
                        else:
                            print(f"{label}: {downloaded / 1024**2:.1f} MiB", flush=True)
                        last_log = now
            break
        except Exception as exc:
            if attempt == 5:
                raise
            print(f"retry {attempt}/5 {label}: {exc}", file=sys.stderr, flush=True)
            time.sleep(5 * attempt)
            resume_from = part_path.stat().st_size if part_path.exists() else 0
            downloaded = resume_from
            headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
            mode = "ab" if resume_from else "wb"

    if expected_size is not None and part_path.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch for {out_path}: got {part_path.stat().st_size}, expected {expected_size}")
    part_path.replace(out_path)
    return f"done {out_path}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download selected HF repositories via hf-mirror.com.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    jobs = []
    for repo, dirname in REPOS:
        target_dir = args.root / dirname
        print(f"listing {repo} -> {target_dir}", flush=True)
        for filename in list_files(repo):
            size = head_size(repo, filename)
            out_path = target_dir / filename
            jobs.append((repo, filename, out_path, size))

    total_known = sum(size for _, _, _, size in jobs if size)
    print(f"files: {len(jobs)}, known bytes: {total_known / 1024**3:.2f} GiB", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_file, *job) for job in jobs]
        for fut in concurrent.futures.as_completed(futures):
            print(fut.result(), flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
