#!/usr/bin/env python3
"""Dependency-free local server for the AMR manual review UI."""

from __future__ import annotations

import json
import os
import tempfile
import csv
import io
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
REVIEWS_PATH = DATA_DIR / "reviews.json"


def read_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def export_rows() -> list:
    manifest = read_json(DATA_DIR / "review_manifest.json", {"cases": []})
    reviews = read_json(REVIEWS_PATH, {})
    rows = []
    for case in manifest.get("cases", []):
        predictions = case.get("pred_relevant_windows", [])
        row = {
            "qid": case.get("qid"),
            "vid": case.get("vid"),
            "query": case.get("query"),
            "gt_windows": json.dumps(case.get("gt_windows", []), ensure_ascii=False),
            "top1_prediction": json.dumps(predictions[0] if predictions else None, ensure_ascii=False),
            "top1_iou": case.get("top1_best_iou"),
            "oracle10_iou": case.get("oracle10_best_iou"),
            "max_gt_length": case.get("max_gt_length"),
        }
        saved = reviews.get(case.get("qid"), {})
        if isinstance(saved, dict):
            row.update(saved)
        rows.append(row)
    return rows


def export_response(format_name: str) -> tuple:
    rows = export_rows()
    fields = ["qid", "vid", "query", "gt_windows", "top1_prediction", "top1_iou", "oracle10_iou", "max_gt_length"]
    review_fields = sorted({key for row in rows for key in row if key not in fields})
    fields.extend(review_fields)
    if format_name == "json":
        body = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
        return "application/json; charset=utf-8", "manual_reviews.json", body
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return "text/csv; charset=utf-8", "manual_reviews.csv", stream.getvalue().encode("utf-8")


class ReviewHandler(SimpleHTTPRequestHandler):
    def _json_response(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/manifest":
            self._json_response(200, read_json(DATA_DIR / "review_manifest.json", {"cases": []}))
            return
        if path == "/api/reviews":
            self._json_response(200, read_json(REVIEWS_PATH, {}))
            return
        if path == "/api/health":
            self._json_response(200, {"status": "ok"})
            return
        if path == "/api/export":
            format_name = parse_qs(urlparse(self.path).query).get("format", ["csv"])[0]
            if format_name not in {"csv", "json"}:
                self._json_response(400, {"error": "format must be csv or json"})
                return
            content_type, filename, body = export_response(format_name)
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Disposition", "attachment; filename={}".format(filename))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/save-review":
            self._json_response(404, {"error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            qid = payload.get("qid")
            review = payload.get("review")
            if not isinstance(qid, str) or not qid or not isinstance(review, dict):
                raise ValueError("qid and review object are required")
            reviews = read_json(REVIEWS_PATH, {})
            if not isinstance(reviews, dict):
                reviews = {}
            previous = reviews.get(qid, {})
            if not isinstance(previous, dict):
                previous = {}
            merged: Dict[str, Any] = dict(previous)
            merged.update(review)
            reviews[qid] = merged
            atomic_write_json(REVIEWS_PATH, reviews)
            self._json_response(200, {"ok": True, "qid": qid, "review": merged})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json_response(400, {"error": str(exc)})
        except OSError as exc:
            self._json_response(500, {"error": str(exc)})


def main() -> None:
    os.chdir(ROOT)
    port = int(os.environ.get("AMR_REVIEW_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), ReviewHandler)
    print("http://127.0.0.1:{}".format(port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
