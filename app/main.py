"""FastAPI application for the Social Media Analytics Dashboard. Author: Rayyan Anwar"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

from . import __author__, __version__
from .analytics import REQUIRED, InputError, analyze, load_posts, report_markdown
from .data import generate_posts

MAX_BYTES = 5_000_000
FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
app = FastAPI(title="Social Media Analytics Dashboard", version=__version__,
              description="Validate and analyse imported post datasets. Author: Rayyan Anwar.",
              contact={"name": __author__}, license_info={"name": "MIT"})


@app.exception_handler(InputError)
async def on_input_error(_: Request, exc: InputError):
    return JSONResponse(status_code=422, content={"error": {"code": "invalid_input", "message": str(exc)}})


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "author": __author__}


@app.get("/api/info")
def info():
    return {"name": "Social Media Analytics Dashboard", "author": __author__, "version": __version__,
            "required_columns": list(REQUIRED), "max_upload_bytes": MAX_BYTES}


@app.get("/api/sample", response_class=PlainTextResponse)
def sample(n: int = Query(2000, ge=10, le=20000), seed: int = Query(0, ge=0, le=10_000)):
    df, _ = generate_posts(n, 60, seed)
    return PlainTextResponse(df.to_csv(index=False), media_type="text/csv")


@app.post("/api/analyze")
async def analyze_endpoint(request: Request):
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="upload too large (limit 5 MB)")
    body = await request.body()
    if len(body) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="upload too large (limit 5 MB)")
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise InputError("file must be UTF-8 encoded CSV") from None

    def work():
        df, rep = load_posts(text)
        a = analyze(df, rep)
        a["markdown"] = report_markdown(a)
        return a
    return await run_in_threadpool(work)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND)
