"""Smart-Log web application.

Lets a user upload a new HDFS log (and optionally a matching ground-truth
labels CSV) and runs it through the already-trained Isolation Forest /
One-Class SVM pipeline (src/inference.py) in a background thread, then
shows a results dashboard: detection summary, alerts, SHAP-derived
explanations, and charts. Each upload gets its own job id and its own
isolated output directory so concurrent/successive runs never collide.

This is a live-inference companion to main.py (which trains the models on
the full labeled research dataset); it does not retrain the models.
"""

import shutil
import threading
import traceback
import uuid
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

from src.inference import InferenceResult, ModelsNotTrainedError, run_inference

PROJECT_ROOT = Path(__file__).parent
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "uploads"
ALLOWED_LOG_EXTENSIONS = {".log", ".txt"}
ALLOWED_LABEL_EXTENSIONS = {".csv"}

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512MB upload cap

_jobs_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def _set_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs[job_id].update(fields)


def _get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _worker(job_id: str, log_path: Path, labels_path: Path | None) -> None:
    def progress(message: str) -> None:
        _set_job(job_id, message=message)

    try:
        _set_job(job_id, status="running", message="Starting...")
        result = run_inference(
            job_id=job_id,
            log_path=log_path,
            output_root=OUTPUT_ROOT,
            labels_path=labels_path,
            progress=progress,
        )
        _set_job(job_id, status="done", result=result, message="Completed")
    except ModelsNotTrainedError as exc:
        _set_job(job_id, status="error", error=str(exc))
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        traceback.print_exc()
        _set_job(job_id, status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        # Uploaded raw input files are no longer needed once scoring is done.
        shutil.rmtree(log_path.parent, ignore_errors=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    log_file = request.files.get("log_file")
    labels_file = request.files.get("labels_file")

    if not log_file or not log_file.filename:
        return render_template("index.html", error="Please choose an HDFS log file to upload."), 400

    log_ext = Path(log_file.filename).suffix.lower()
    if log_ext not in ALLOWED_LOG_EXTENSIONS:
        return render_template("index.html", error="Log file must be a .log or .txt file."), 400

    job_id = uuid.uuid4().hex[:12]
    job_input_dir = UPLOADS_DIR / job_id
    job_input_dir.mkdir(parents=True, exist_ok=True)

    log_path = job_input_dir / "upload.log"
    log_file.save(log_path)

    labels_path = None
    if labels_file and labels_file.filename:
        labels_ext = Path(labels_file.filename).suffix.lower()
        if labels_ext not in ALLOWED_LABEL_EXTENSIONS:
            shutil.rmtree(job_input_dir, ignore_errors=True)
            return render_template("index.html", error="Labels file must be a .csv file."), 400
        labels_path = job_input_dir / "labels.csv"
        labels_file.save(labels_path)

    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "message": "Queued...", "result": None, "error": None}

    thread = threading.Thread(target=_worker, args=(job_id, log_path, labels_path), daemon=True)
    thread.start()

    return redirect(url_for("job_status", job_id=job_id))


@app.route("/jobs/<job_id>")
def job_status(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return render_template("index.html", error="Unknown job id."), 404
    if job["status"] == "done":
        return redirect(url_for("job_results", job_id=job_id))
    return render_template("job_status.html", job_id=job_id, job=job)


@app.route("/api/jobs/<job_id>")
def api_job_status(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return jsonify({"status": "unknown"}), 404
    return jsonify({"status": job["status"], "message": job.get("message", ""), "error": job.get("error")})


def _severity_rank(sev: str) -> int:
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    return order.get(sev, 4)


@app.route("/jobs/<job_id>/results")
def job_results(job_id: str):
    job = _get_job(job_id)
    if job is None:
        return render_template("index.html", error="Unknown job id."), 404
    if job["status"] == "error":
        return render_template("job_status.html", job_id=job_id, job=job), 200
    if job["status"] != "done":
        return redirect(url_for("job_status", job_id=job_id))

    result: InferenceResult = job["result"]
    alerts_sorted = sorted(result.alerts, key=lambda a: _severity_rank(a.severity))
    page_size = 200
    shown_alerts = alerts_sorted[:page_size]

    comparison_records = None
    if result.comparison_table is not None:
        comparison_records = result.comparison_table.round(4).to_dict(orient="records")

    return render_template(
        "results.html",
        job_id=job_id,
        result=result,
        comparison_records=comparison_records,
        shown_alerts=shown_alerts,
        total_alerts=len(alerts_sorted),
        page_size=page_size,
    )


@app.route("/jobs/<job_id>/plots/<filename>")
def job_plot(job_id: str, filename: str):
    plot_path = OUTPUT_ROOT / job_id / "plots" / filename
    if not plot_path.is_file():
        return "Not found", 404
    return send_file(plot_path)


@app.route("/jobs/<job_id>/download/<kind>")
def job_download(job_id: str, kind: str):
    job = _get_job(job_id)
    if job is None or job["status"] != "done":
        return "Not found", 404
    result: InferenceResult = job["result"]
    if kind == "csv":
        return send_file(result.alerts_csv_path, as_attachment=True, download_name="alerts.csv")
    if kind == "json":
        return send_file(result.alerts_json_path, as_attachment=True, download_name="alerts.json")
    return "Not found", 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
