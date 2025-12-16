from __future__ import annotations

import argparse
import datetime as dt
import html
import os
import re
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse


@dataclass
class Job:
    id: str
    mode: str
    input_dir: Path
    output_dir: Path
    verbose: bool
    source: str = "path"  # path|upload
    uploaded_files: int = 0
    status: str = "queued"  # queued|running|done|error
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().is_relative_to(root.resolve())
        return True
    except Exception:
        return False


def _tail(path: Path, max_lines: int = 200) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-max_lines:])

def _safe_filename(name: str) -> str:
    base = Path(name).name
    base = base.strip().replace("\x00", "")
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return base or "upload.dcm"


def _html_page(title: str, body: str) -> HTMLResponse:
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }}
    .row {{ margin: 12px 0; }}
    label {{ display: inline-block; min-width: 120px; }}
    input[type=text] {{ width: min(900px, 95vw); padding: 8px; }}
    select {{ padding: 8px; }}
    button {{ padding: 10px 14px; }}
    pre {{ background: #0b1020; color: #e8e8e8; padding: 12px; overflow: auto; border-radius: 8px; }}
    .muted {{ color: #666; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
    @media (min-width: 1000px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
  </style>
</head>
<body>
{body}
</body>
</html>"""
    )


def create_app(*, data_root: Optional[Path] = None) -> FastAPI:
    app = FastAPI(title="MENN WebUI", version="0.1")
    jobs: dict[str, Job] = {}
    jobs_lock = threading.Lock()
    upload_root = (
        Path(os.environ["MENN_UPLOAD_ROOT"]).expanduser().resolve()
        if os.environ.get("MENN_UPLOAD_ROOT")
        else ((data_root / "uploads") if data_root else (Path.cwd() / "uploads"))
    )

    def validate_input_dir(input_dir: Path) -> None:
        if not input_dir.exists() or not input_dir.is_dir():
            raise HTTPException(status_code=400, detail=f"input_dir does not exist or is not a directory: {input_dir}")
        if data_root and not _within(input_dir, data_root):
            raise HTTPException(status_code=400, detail=f"input_dir must be under {data_root}")

    def validate_output_dir(output_dir: Path) -> None:
        if data_root and not _within(output_dir, data_root):
            raise HTTPException(status_code=400, detail=f"output_dir must be under {data_root}")

    def run_job(job: Job) -> None:
        job.started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        job.status = "running"
        try:
            if job.mode == "auto":
                from menn_cli import main as menn_main

                menn_main(
                    [
                        "auto",
                        "--input-dir",
                        str(job.input_dir),
                        "--output-dir",
                        str(job.output_dir),
                        "--verbose" if job.verbose else "--no-verbose",
                    ]
                )
            elif job.mode == "bmode":
                from echoanalysis_main import run_bmode_analysis

                run_bmode_analysis(
                    input_dir=str(job.input_dir),
                    output_dir=str(job.output_dir),
                    verbose=job.verbose,
                    weights=None,
                )
            elif job.mode == "mmode":
                from echoanalysis_mmode_main import run_mmode_analysis

                run_mmode_analysis(
                    input_dir=str(job.input_dir),
                    output_dir=str(job.output_dir),
                    verbose=job.verbose,
                    weights=None,
                )
            else:
                raise RuntimeError(f"Unknown mode: {job.mode}")
            job.status = "done"
        except BaseException as e:
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
        finally:
            job.finished_at = dt.datetime.now(dt.timezone.utc).isoformat()

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        root_note = f"Data root restriction: {data_root}" if data_root else "No data root restriction."
        body = f"""
<h1>MENN WebUI</h1>
<p class="muted">{html.escape(root_note)}</p>
<h2>Run (from path)</h2>
<form method="post" action="/run">
  <div class="row">
    <label for="mode">Mode</label>
    <select id="mode" name="mode">
      <option value="auto" selected>Auto (B-Mode + M-Mode)</option>
      <option value="bmode">B-Mode only</option>
      <option value="mmode">M-Mode only</option>
    </select>
  </div>
  <div class="row">
    <label for="input_dir">Input dir</label>
    <input id="input_dir" name="input_dir" type="text" placeholder="/data/study1"/>
  </div>
  <div class="row">
    <label for="output_dir">Output dir</label>
    <input id="output_dir" name="output_dir" type="text" placeholder="(optional) defaults to &lt;input&gt;/output"/>
  </div>
  <div class="row">
    <label for="verbose">QC outputs</label>
    <input id="verbose" name="verbose" type="checkbox" checked/>
    <span class="muted">Write annotated PDFs</span>
  </div>
  <div class="row">
    <button type="submit">Run</button>
  </div>
</form>

<h2>Run (upload files)</h2>
<form method="post" action="/run-upload" enctype="multipart/form-data">
  <div class="row">
    <label for="mode_u">Mode</label>
    <select id="mode_u" name="mode">
      <option value="auto" selected>Auto (B-Mode + M-Mode)</option>
      <option value="bmode">B-Mode only</option>
      <option value="mmode">M-Mode only</option>
    </select>
  </div>
  <div class="row">
    <label for="files">DICOM files</label>
    <input id="files" name="files" type="file" multiple accept=".dcm" webkitdirectory directory/>
    <span class="muted">Select multiple files or a folder (supported in Chrome/Edge).</span>
  </div>
  <div class="row">
    <label for="verbose_u">QC outputs</label>
    <input id="verbose_u" name="verbose" type="checkbox" checked/>
    <span class="muted">Write annotated PDFs</span>
  </div>
  <div class="row">
    <button type="submit">Upload &amp; Run</button>
  </div>
  <p class="muted">Uploads are stored under: {html.escape(str(upload_root))}</p>
</form>

<h2>Jobs</h2>
<ul>
"""
        with jobs_lock:
            for job_id, job in list(jobs.items())[-25:][::-1]:
                body += (
                    f'<li><a href="/jobs/{job_id}">{job_id}</a> — {html.escape(job.status)}'
                    f' — {html.escape(job.mode)} — {html.escape(job.source)} — {html.escape(str(job.input_dir))}</li>'
                )
        body += "</ul>"
        return _html_page("MENN WebUI", body)

    @app.post("/run")
    def start_run(
        mode: str = Form(...),
        input_dir: str = Form(...),
        output_dir: str = Form(""),
        verbose: Optional[str] = Form(None),
    ) -> RedirectResponse:
        mode = mode.strip().lower()
        if mode not in {"auto", "bmode", "mmode"}:
            raise HTTPException(status_code=400, detail="mode must be one of: auto, bmode, mmode")

        input_path = Path(input_dir).expanduser()
        validate_input_dir(input_path)

        if output_dir.strip():
            output_path = Path(output_dir).expanduser()
        else:
            output_path = input_path / "output"
        validate_output_dir(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        job = Job(
            id=str(uuid.uuid4())[:8],
            mode=mode,
            input_dir=input_path.resolve(),
            output_dir=output_path.resolve(),
            verbose=bool(verbose),
            source="path",
        )

        with jobs_lock:
            jobs[job.id] = job

        t = threading.Thread(target=run_job, args=(job,), daemon=True)
        t.start()

        return RedirectResponse(url=f"/jobs/{job.id}", status_code=303)

    @app.post("/run-upload")
    async def start_upload_run(
        mode: str = Form(...),
        verbose: Optional[str] = Form(None),
        files: list[UploadFile] = File(...),
    ) -> RedirectResponse:
        mode = mode.strip().lower()
        if mode not in {"auto", "bmode", "mmode"}:
            raise HTTPException(status_code=400, detail="mode must be one of: auto, bmode, mmode")
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")

        upload_root.mkdir(parents=True, exist_ok=True)
        if data_root and not _within(upload_root, data_root):
            raise HTTPException(status_code=500, detail=f"upload root must be under {data_root}")

        job_id = str(uuid.uuid4())[:8]
        input_path = (upload_root / job_id / "input").resolve()
        output_path = (upload_root / job_id / "output").resolve()
        input_path.mkdir(parents=True, exist_ok=True)
        output_path.mkdir(parents=True, exist_ok=True)

        uploaded = 0
        for uf in files:
            name = _safe_filename(uf.filename or "upload.dcm")
            if Path(name).suffix.lower() != ".dcm":
                continue
            target = input_path / name
            if target.exists():
                target = input_path / f"{target.stem}_{uploaded}{target.suffix}"
            with target.open("wb") as f:
                shutil.copyfileobj(uf.file, f)
            uploaded += 1

        if uploaded == 0:
            raise HTTPException(status_code=400, detail="No .dcm files uploaded")

        job = Job(
            id=job_id,
            mode=mode,
            input_dir=input_path,
            output_dir=output_path,
            verbose=bool(verbose),
            source="upload",
            uploaded_files=uploaded,
        )
        with jobs_lock:
            jobs[job.id] = job

        t = threading.Thread(target=run_job, args=(job,), daemon=True)
        t.start()
        return RedirectResponse(url=f"/jobs/{job.id}", status_code=303)

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_page(job_id: str) -> HTMLResponse:
        with jobs_lock:
            job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")

        auto_run_log = job.output_dir / "run.log"
        auto_errors_log = job.output_dir / "errors.log"
        mode_run_log = job.output_dir / ("B-Mode/run.log" if job.mode == "auto" else "run.log")
        mode_errors_log = job.output_dir / ("B-Mode/errors.log" if job.mode == "auto" else "errors.log")

        default_runlog_name = "auto-run" if job.mode == "auto" else "run"
        default_errlog_name = "auto-errors" if job.mode == "auto" else "errors"

        body = f"""
<h1>Job {html.escape(job.id)}</h1>
<p><a href="/">Back</a></p>
<div class="row"><b>Status:</b> <span id="status">{html.escape(job.status)}</span></div>
<div class="row"><b>Mode:</b> {html.escape(job.mode)}</div>
<div class="row"><b>Source:</b> {html.escape(job.source)} {('(files=' + str(job.uploaded_files) + ')' ) if job.source == 'upload' else ''}</div>
<div class="row"><b>Input:</b> {html.escape(str(job.input_dir))}</div>
<div class="row"><b>Output:</b> {html.escape(str(job.output_dir))}</div>
<div class="row"><b>QC outputs:</b> {str(job.verbose).lower()}</div>
<div class="row"><b>Created:</b> {html.escape(job.created_at)}</div>
<div class="row"><b>Started:</b> {html.escape(job.started_at or '')}</div>
<div class="row"><b>Finished:</b> {html.escape(job.finished_at or '')}</div>
<div class="row"><b>Error:</b> {html.escape(job.error or '')}</div>

<div class="grid">
  <div>
    <h2>Run log</h2>
    <pre id="runlog">{html.escape(_tail(auto_run_log if job.mode == "auto" else mode_run_log))}</pre>
    <div class="muted">Showing: {html.escape(default_runlog_name)}</div>
  </div>
  <div>
    <h2>Errors</h2>
    <pre id="errlog">{html.escape(_tail(auto_errors_log if job.mode == "auto" else mode_errors_log))}</pre>
    <div class="muted">Showing: {html.escape(default_errlog_name)}</div>
  </div>
</div>

<h2>Output files</h2>
<ul id="files"></ul>

<script>
async function refresh() {{
  const r = await fetch('/api/jobs/{job_id}');
  if (!r.ok) return;
  const j = await r.json();
  document.getElementById('status').textContent = j.status;

  const runlog = await fetch('/api/jobs/{job_id}/log?name={default_runlog_name}');
  if (runlog.ok) document.getElementById('runlog').textContent = await runlog.text();
  const errlog = await fetch('/api/jobs/{job_id}/log?name={default_errlog_name}');
  if (errlog.ok) document.getElementById('errlog').textContent = await errlog.text();

  const files = await fetch('/api/jobs/{job_id}/files');
  if (files.ok) {{
    const arr = await files.json();
    const ul = document.getElementById('files');
    ul.innerHTML = '';
    for (const f of arr) {{
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = '/api/jobs/{job_id}/download?rel=' + encodeURIComponent(f.rel);
      a.textContent = f.rel;
      li.appendChild(a);
      ul.appendChild(li);
    }}
  }}
}}
setInterval(refresh, 1500);
refresh();
</script>
"""
        return _html_page(f"Job {job.id}", body)

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, Any]:
        with jobs_lock:
            job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return {
            "id": job.id,
            "mode": job.mode,
            "input_dir": str(job.input_dir),
            "output_dir": str(job.output_dir),
            "verbose": job.verbose,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error": job.error,
        }

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def job_log(job_id: str, name: str = "run", lines: int = 200) -> str:
        with jobs_lock:
            job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if lines < 10 or lines > 2000:
            raise HTTPException(status_code=400, detail="lines must be between 10 and 2000")

        if job.mode == "auto":
            base = job.output_dir / "B-Mode"
        else:
            base = job.output_dir

        if name == "run":
            path = base / "run.log"
        elif name == "errors":
            path = base / "errors.log"
        elif name == "auto-run":
            path = job.output_dir / "run.log"
        elif name == "auto-errors":
            path = job.output_dir / "errors.log"
        else:
            raise HTTPException(status_code=400, detail="unknown log name")

        if data_root and not _within(path, data_root):
            raise HTTPException(status_code=400, detail="log path outside allowed root")
        return _tail(path, max_lines=lines)

    @app.get("/api/jobs/{job_id}/files")
    def job_files(job_id: str) -> list[dict[str, str]]:
        with jobs_lock:
            job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")

        files: list[Path] = []
        for p in job.output_dir.rglob("*"):
            if p.is_file() and p.name.lower() in {"output.csv", "output_images.pdf", "run.log", "errors.log"}:
                files.append(p)

        out = []
        for p in sorted(files):
            rel = str(p.relative_to(job.output_dir))
            out.append({"rel": rel})
        return out

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str, rel: str) -> FileResponse:
        with jobs_lock:
            job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            raise HTTPException(status_code=400, detail="invalid rel path")
        target = (job.output_dir / rel_path).resolve()
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        if data_root and not _within(target, data_root):
            raise HTTPException(status_code=400, detail="file outside allowed root")
        return FileResponse(path=str(target), filename=target.name)

    return app


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="MENN Web UI server (container-friendly).")
    parser.add_argument("--host", default=os.environ.get("MENN_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MENN_PORT", "8000")))
    parser.add_argument(
        "--data-root",
        default=os.environ.get("MENN_DATA_ROOT", ""),
        help="Optional: restrict input/output paths to be under this directory",
    )
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev only)")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else None

    # Reduce TensorFlow log noise by default for the web server process.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import uvicorn

    uvicorn.run(
        create_app(data_root=data_root),
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
