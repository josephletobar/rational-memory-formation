#!/usr/bin/env python3
"""Resumably download the full EgoLongQA validation video set to the external drive."""

from __future__ import annotations

import shutil
import os
import time
from pathlib import Path

from huggingface_hub import HfApi, HfFolder, hf_hub_download


REPO_ID = "facebook/wearable-ai"
SOURCE_DIR = "egolongqa/val"
ANNOTATION = "egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
TARGET = Path("/Volumes/Crucial X9/theory-of-mind")
MIN_FREE_BYTES = 5 * 2**30


def auth_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    token = token or HfFolder.get_token()
    if not token:
        raise RuntimeError(
            "Hugging Face authentication is required. Run `huggingface-cli login` "
            "or start this script with `HF_TOKEN=...`."
        )
    return token


def repo_files(token: str) -> dict[str, int]:
    api = HfApi(token=token)
    tree = api.list_repo_tree(
        REPO_ID,
        path_in_repo=SOURCE_DIR,
        repo_type="dataset",
        expand=True,
    )
    sizes = {
        Path(item.path).name: item.size
        for item in tree
        if item.path.endswith(".mp4")
    }
    if len(sizes) != 700:
        raise RuntimeError(f"Expected 700 videos, found {len(sizes)}")
    return sizes


def existing_file(name: str, expected_size: int) -> Path | None:
    candidates = [
        TARGET / name,
        TARGET / SOURCE_DIR / name,
    ]
    for path in candidates:
        if path.exists() and path.stat().st_size == expected_size:
            return path
    return None


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    token = auth_token()
    sizes = repo_files(token)

    pending = []
    for name, expected_size in sorted(sizes.items()):
        valid = existing_file(name, expected_size)
        if valid is not None:
            if valid.parent != TARGET:
                valid.replace(TARGET / name)
            continue

        destination = TARGET / name
        if destination.exists():
            backup = destination.with_name(f"{destination.name}.invalid-{int(time.time())}")
            destination.replace(backup)
            print(f"Preserved size-mismatched file as {backup.name}", flush=True)
        pending.append(name)

    pending_bytes = sum(sizes[name] for name in pending)
    free_bytes = shutil.disk_usage(TARGET).free
    print(
        f"Total: {len(sizes)}; already valid: {len(sizes) - len(pending)}; "
        f"pending: {len(pending)} ({pending_bytes / 2**30:.2f} GiB); "
        f"free: {free_bytes / 2**30:.2f} GiB",
        flush=True,
    )
    if free_bytes - pending_bytes < MIN_FREE_BYTES:
        raise RuntimeError("Download would leave less than 5 GiB free")

    annotation_path = Path(
        hf_hub_download(
            REPO_ID,
            filename=ANNOTATION,
            repo_type="dataset",
            local_dir=TARGET,
            token=token,
        )
    )
    annotation_path.replace(TARGET / Path(ANNOTATION).name)

    failures: list[str] = []
    for index, name in enumerate(pending, 1):
        print(f"[{index}/{len(pending)}] downloading {name}", flush=True)
        remote_name = f"{SOURCE_DIR}/{name}"
        for attempt in range(1, 4):
            try:
                downloaded = Path(
                    hf_hub_download(
                        REPO_ID,
                        filename=remote_name,
                        repo_type="dataset",
                        local_dir=TARGET,
                        token=token,
                    )
                )
                actual_size = downloaded.stat().st_size
                if actual_size != sizes[name]:
                    raise IOError(f"size mismatch: {actual_size} != {sizes[name]}")
                downloaded.replace(TARGET / name)
                break
            except Exception as exc:
                print(f"  attempt {attempt}/3 failed: {exc}", flush=True)
                if attempt == 3:
                    failures.append(name)
                else:
                    time.sleep(5 * attempt)
        if name not in failures:
            remaining = sum(sizes[n] for n in pending[index:])
            print(f"  saved; approximately {remaining / 2**30:.2f} GiB remaining", flush=True)

    if failures:
        print(f"FAILED ({len(failures)}): {', '.join(failures)}", flush=True)
        raise SystemExit(1)
    print("DOWNLOAD COMPLETE: all 700 EgoLongQA validation videos validated", flush=True)


if __name__ == "__main__":
    main()
