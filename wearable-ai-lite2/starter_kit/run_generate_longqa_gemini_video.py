#!/usr/bin/env python3
"""Generate LongQA predictions by sending raw MP4 files to Gemini.

This script is intentionally separate from the main frame-sampling pipeline and
does not use OpenCV or image extraction. Each sample uploads the full MP4 to the
Gemini Files API and passes that file URI through generateContent.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from run_generate_longqa import LONGQA_PROMPT_TEMPLATE, _first_existing, _resolve_path

logger = logging.getLogger(__name__)


X9_LONGQA_ROOT = "/Volumes/Crucial X9/theory-of-mind"

DEFAULT_INPUT = _first_existing(
    os.path.join(X9_LONGQA_ROOT, "wearable_ai_2026_egolongqa_val_700.jsonl"),
    "../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl",
)
DEFAULT_VIDEO_FOLDER = _first_existing(
    os.path.join(X9_LONGQA_ROOT, "egolongqa_merged_val"),
    os.path.join(X9_LONGQA_ROOT, "egolongqa/val"),
    "../egolongqa/val",
)
DEFAULT_OUTPUT = "output/egolongqa/predictions_gemini_video.jsonl"


def _load_dotenv(env_path: str | None = None) -> None:
    path = env_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".env"
    )
    if not os.path.exists(path):
        return
    try:
        with open(path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if (len(value) >= 2) and (
                    (value[0] == value[-1] == '"')
                    or (value[0] == value[-1] == "'")
                ):
                    value = value[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        logger.warning("Could not read .env file at %s", path)


def load_jsonl(path: str) -> list[dict[str, object]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


class GeminiVideoClient:
    """Small wrapper around Gemini REST endpoints for file ingestion + generation."""

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash-lite",
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str = "v1beta",
        timeout: int = 1200,
        poll_interval: float = 2.0,
        poll_timeout: int = 120,
    ) -> None:
        self.model_id = self._normalize_model_id(model_id)
        _load_dotenv()
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GOOGLE_GENERATIVE_AI_API_KEY")
        )
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Add GEMINI_API_KEY in starter_kit/.env."
            )

        base = (api_base or os.environ.get("GEMINI_API_BASE", "")).rstrip("/")
        self.api_base = base or "https://generativelanguage.googleapis.com"
        self.api_version = (api_version or os.environ.get("GEMINI_API_VERSION", "v1beta")).strip(
            "/"
        )
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.endpoint = f"{self.api_base.rstrip('/')}/{self.api_version}"

    @staticmethod
    def _normalize_model_id(model_id: str) -> str:
        return model_id if model_id.startswith("models/") else f"models/{model_id}"

    def _api_url(self, path: str, query: bool = True) -> str:
        base = f"{self.endpoint}/{path.lstrip('/')}"
        if query:
            return f"{base}?{urllib.parse.urlencode({'key': self.api_key})}"
        return base

    def _upload_session_url(self) -> str:
        return f"{self.api_base.rstrip('/')}/upload/v1beta/files?{urllib.parse.urlencode({'key': self.api_key})}"

    @staticmethod
    def _mime_type(video_path: str) -> str:
        suffix = Path(video_path).suffix.lower()
        if suffix in {".mp4", ".m4v"}:
            return "video/mp4"
        if suffix == ".mov":
            return "video/quicktime"
        if suffix == ".avi":
            return "video/x-msvideo"
        if suffix == ".mkv":
            return "video/x-matroska"
        return "video/mp4"

    def _request_json(
        self, req: urllib.request.Request, timeout: int | None = None
    ) -> dict[str, object]:
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                payload = json.loads(resp.read())
                return payload
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            raise RuntimeError(f"Gemini API call failed ({e.code}): {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Gemini API request failed: {e}") from e

    def upload_video(self, video_path: str) -> str:
        """Upload a local video and return the resulting Gemini file URI."""
        size = os.path.getsize(video_path)
        mime_type = self._mime_type(video_path)
        display_name = Path(video_path).name

        start_headers = {
            "Content-Type": "application/json; charset=utf-8",
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
        }
        start_body = json.dumps({"file": {"display_name": display_name}}).encode(
            "utf-8"
        )
        start_req = urllib.request.Request(
            self._upload_session_url(),
            data=start_body,
            headers=start_headers,
            method="POST",
        )
        start_resp: dict[str, object] | None = None
        with urllib.request.urlopen(start_req, timeout=self.timeout) as resp:
            start_resp = json.loads(resp.read() or b"{}")
            upload_url = resp.getheader("x-goog-upload-url")
        if not upload_url:
            raise RuntimeError(
                f"Gemini upload start did not return x-goog-upload-url. response={start_resp}"
            )

        with open(video_path, "rb") as f:
            video_bytes = f.read()
        upload_headers = {
            "Content-Type": mime_type,
            "X-Goog-Upload-Command": "upload, finalize",
            "X-Goog-Upload-Offset": "0",
            "Content-Length": str(len(video_bytes)),
        }
        upload_req = urllib.request.Request(
            upload_url,
            data=video_bytes,
            headers=upload_headers,
            method="POST",
        )
        upload_result = self._request_json(upload_req)
        file_payload = upload_result.get("file") if isinstance(upload_result, dict) else None
        if isinstance(file_payload, dict):
            file_name = file_payload.get("uri") or file_payload.get("name")
            if file_name:
                return str(file_name)
        if "name" in upload_result:
            return str(upload_result["name"])
        if "uri" in upload_result:
            return str(upload_result["uri"])
        raise RuntimeError(f"Could not parse upload response: {upload_result}")

    def wait_until_active(self, file_uri: str) -> None:
        if file_uri.startswith("http://") or file_uri.startswith("https://"):
            file_id = file_uri.split("/v1beta/", 1)[-1]
        else:
            file_id = file_uri
        if file_id.startswith("/"):
            file_id = file_id[1:]

        deadline = time.time() + self.poll_timeout
        last_state = None
        while time.time() < deadline:
            req = urllib.request.Request(
                self._api_url(file_id, query=True),
                method="GET",
            )
            state_payload = self._request_json(req)
            state = state_payload.get("state")
            if state:
                last_state = state
            if state == "ACTIVE":
                return
            if state == "FAILED":
                error = state_payload.get("error")
                raise RuntimeError(f"Gemini file processing failed ({file_uri}): {error}")
            if state in {None, "PROCESSING", "UPLOADING"}:
                time.sleep(self.poll_interval)
                continue
            raise RuntimeError(
                f"Unexpected Gemini file state for {file_uri}: {state} (payload={state_payload})"
            )
        raise RuntimeError(
            f"Timed out waiting for file {file_uri} to become ACTIVE (last_state={last_state})"
        )

    def predict(self, file_uri: str, prompt: str, max_tokens: int = 16) -> str:
        if file_uri.startswith("https://generativelanguage.googleapis.com/"):
            file_ref = file_uri
        elif file_uri.startswith("files/"):
            file_ref = file_uri
        else:
            file_ref = file_uri

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "file_data": {
                                "mime_type": self._mime_type(file_ref),
                                "file_uri": file_ref,
                            }
                        },
                        {"text": prompt},
                    ],
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.0,
            },
        }

        req = urllib.request.Request(
            self._api_url(f"{self.model_id}:generateContent"),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        result = self._request_json(req)
        if "error" in result:
            raise RuntimeError(f"Gemini generateContent error: {result['error']}")

        candidates = result.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"No candidates in Gemini response: {result}")
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(
            part.get("text", "") for part in parts if isinstance(part, dict)
        ).strip()
        if text:
            return text
        if "text" in candidates[0].get("content", {}):
            return str(candidates[0]["content"]["text"]).strip()
        raise RuntimeError(f"Empty Gemini text response: {result}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run LongQA directly on MP4 files with Gemini's file upload API "
            "(no frame extraction)."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="Input JSONL file (video_path, question, mcq_options).",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default=DEFAULT_VIDEO_FOLDER,
        help="Folder with raw .mp4 files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Prediction output JSONL path.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gemini-3.5-flash",
        help="Gemini model id (for example gemini-3.5-flash).",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key override (otherwise reads GEMINI_API_KEY).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Process only the first N samples (defaults to 1).",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between file processing status checks.",
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=180,
        help="Max seconds to wait for each file to become ACTIVE.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
        help="Maximum output tokens from Gemini.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    video_folder = _resolve_path(args.video_folder)

    data = load_jsonl(input_path)
    if args.max_samples is not None:
        data = data[: args.max_samples]

    client = GeminiVideoClient(
        model_id=args.model_id,
        api_key=args.gemini_api_key,
        poll_interval=args.poll_interval,
        poll_timeout=args.poll_timeout,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as out_f:
        for i, row in enumerate(data, start=1):
            video_path = os.path.join(video_folder, str(row["video_path"]))
            prompt = LONGQA_PROMPT_TEMPLATE.format(
                question=row["question"],
                mcq_options=row["mcq_options"],
            )
            file_uri = client.upload_video(video_path)
            client.wait_until_active(file_uri)
            response = client.predict(file_uri, prompt, max_tokens=args.max_output_tokens)
            pred = dict(row)
            pred["mcq_answer"] = response
            pred["gemini_file_uri"] = file_uri
            out_f.write(json.dumps(pred) + "\n")
            out_f.flush()
            print(f"[{i}/{len(data)}] {Path(video_path).name}: {response}")

    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
