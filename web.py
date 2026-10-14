#!/usr/bin/env -S uv run --project /home/jt/claude/clipsmith python
"""clipsmith web app: pick or upload a video (or paste a URL), watch it work, then review, edit and
re-render the shorts. Local only (127.0.0.1). One GPU job at a time, in a queue.

    ./serve.sh            # http://127.0.0.1:8765
"""
import os
import queue
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

import clipsmith as cs

HERE = Path(__file__).resolve().parent
CLIPS = HERE / "clips"
UPLOADS = HERE / "uploads"
LIBRARY = [Path(p).expanduser() for p in os.environ.get("CLIPSMITH_LIBRARY", "~/Videos").split(":")] + \
          [UPLOADS, HERE / "downloads"]
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}

app = FastAPI(title="clipsmith")
JOBS = {}            # id -> job dict
Q = queue.Queue()


# ---------------------------------------------------------------- job queue (one GPU job at a time)

def new_job(kind, project, fn, label):
    j = {"id": uuid.uuid4().hex[:8], "kind": kind, "project": project, "label": label, "status": "queued",
         "progress": 0.0, "log": [], "error": None, "created": time.time(), "fn": fn}
    JOBS[j["id"]] = j
    Q.put(j)
    return public(j)


def public(j):
    return {k: v for k, v in j.items() if k != "fn"} | {"log": j["log"][-40:]}


def busy(slug):
    return any(j["project"] == slug and j["status"] in ("queued", "running") for j in JOBS.values())


def worker():
    while True:
        j = Q.get()
        j["status"], j["started"] = "running", time.time()

        def hook(msg, progress, j=j):
            j["log"].append(msg)
            if progress is not None:
                j["progress"] = progress
        cs._hooks.append(hook)
        try:
            j["fn"](j)
            j["status"], j["progress"] = "done", 1.0
        except SystemExit as e:          # clipsmith's friendly errors (GPU full, no speech, ...)
            j["status"], j["error"] = "error", str(e)
        except Exception as e:  # noqa: BLE001
            j["status"], j["error"] = "error", f"{e}"
            j["log"].append(traceback.format_exc()[-2000:])
        finally:
            cs._hooks.remove(hook)
            j["finished"] = time.time()


threading.Thread(target=worker, daemon=True).start()


# ---------------------------------------------------------------- helpers

def project(slug):
    p = cs.Project(CLIPS / slug)
    if not p.file.exists() or "/" in slug or slug.startswith("."):
        raise HTTPException(404, "no such project")
    return p


def summary(p):
    d = p.data
    clips = sorted(d.get("clips", []), key=lambda c: -c.get("score", 0))
    thumb = next((c["file"] for c in clips if c.get("file")), None)
    return {"slug": p.dir.name, "video": Path(d.get("video", "")).name, "duration": d.get("info", {}).get("duration"),
            "clips": len(clips), "rendered": sum(1 for c in clips if c.get("file")), "created": d.get("created"),
            "thumb": f"/files/{p.dir.name}/{Path(thumb).with_suffix('.jpg').name}" if thumb else None,
            "busy": busy(p.dir.name)}


# ---------------------------------------------------------------- API

@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/meta")
def meta():
    brand = cs.load_brand(HERE / "brand.json")
    return {"defaults": {**cs.Settings().to_dict(), **brand}, "captions": list(cs.CAPTION_STYLES) + ["none"],
            "aspects": list(cs.ASPECTS), "layouts": ["auto", "track", "split", "fit", "center"]}


@app.get("/api/state")
def state():
    projects = []
    for f in sorted(CLIPS.glob("*/project.json"), key=lambda f: -f.stat().st_mtime):
        try:
            projects.append(summary(cs.Project(f.parent)))
        except Exception:  # noqa: BLE001
            pass
    jobs = sorted(JOBS.values(), key=lambda j: -j["created"])[:20]
    return {"projects": projects, "jobs": [public(j) for j in jobs]}


@app.get("/api/library")
def library():
    out = []
    for root in LIBRARY:
        if not root.exists():
            continue
        for f in root.rglob("*"):
            if f.suffix.lower() in VIDEO_EXT and f.is_file() and ".work" not in f.parts and CLIPS not in f.parents:
                out.append({"path": str(f), "name": f.name, "folder": str(f.parent).replace(str(Path.home()), "~"),
                            "size": f.stat().st_size, "mtime": f.stat().st_mtime,
                            "project": (CLIPS / cs.slug(f.stem) / "project.json").exists()})
    return sorted(out, key=lambda x: -x["mtime"])


@app.post("/api/upload")
async def upload(file: UploadFile):
    UPLOADS.mkdir(exist_ok=True)
    name = Path(file.filename or "upload.mp4").name
    dest = UPLOADS / name
    with dest.open("wb") as f:
        while chunk := await file.read(8 << 20):
            f.write(chunk)
    return {"path": str(dest)}


@app.post("/api/projects")
async def create(req: Request):
    body = await req.json()
    source = (body.get("source") or "").strip()
    if not source:
        raise HTTPException(400, "pick a video or paste a URL")
    s = cs.Settings.from_dict({**meta()["defaults"], **(body.get("settings") or {})})

    def run(j):
        p = cs.create_project(source, CLIPS, s, reselect=bool(body.get("reselect")))
        j["project"] = p.dir.name
        cs.render_all(p)
    label = Path(source).name if "://" not in source else source
    return new_job("create", cs.slug(Path(source).stem) if "://" not in source else None, run, f"New clips: {label}")


@app.get("/api/projects/{slug}")
def get_project(slug: str):
    p = project(slug)
    return {**p.data, "slug": slug, "busy": busy(slug)}


@app.get("/api/projects/{slug}/transcript")
def transcript(slug: str):
    p = project(slug)
    return [{"i": i, "s": x["s"], "e": x["e"], "text": x["text"]} for i, x in enumerate(cs.sentences_from_words(p.words))]


def editable(slug):
    if busy(slug):
        raise HTTPException(409, "this project is rendering; wait for it to finish")
    return project(slug)


@app.patch("/api/projects/{slug}/clips/{cid}")
async def edit_clip(slug: str, cid: int, req: Request):
    body = await req.json()
    p = editable(slug)
    try:
        c = p.clip(cid)
    except StopIteration:
        raise HTTPException(404, "no such clip")
    for k in ("title", "caption", "hashtags", "keywords"):
        if k in body:
            c[k] = body[k]
    if "start" in body or "end" in body:
        try:
            p.set_range(c, body.get("start", c["start"]), body.get("end", c["end"]))
        except ValueError as e:
            raise HTTPException(400, str(e))
    if "overrides" in body:
        c["overrides"] = {k: v for k, v in body["overrides"].items() if v not in (None, "")}
    p.save()
    if body.get("render", True):
        return new_job("render", slug, lambda j: render_ids(slug, [cid]), f"Re-render: {c.get('title', '')}")
    return {"ok": True}


def render_ids(slug, ids, settings=None):
    p = project(slug)
    if settings:
        p.data["settings"] = {**p.data.get("settings", {}), **settings}
        p.save()
    cs.render_all(p, only=set(ids) if ids else None)


@app.post("/api/projects/{slug}/clips")
async def add_clip(slug: str, req: Request):
    body = await req.json()
    p = editable(slug)
    c = {"id": p.next_id(), "score": 0, "scores": {}, "title": body.get("title") or "New clip", "caption": "",
         "hashtags": [], "keywords": [], "why": "picked by hand"}
    try:
        p.set_range(c, body["start"], body["end"])
    except (KeyError, ValueError) as e:
        raise HTTPException(400, f"bad range: {e}")
    if c["i0"] is not None and not body.get("title"):
        c["title"] = " ".join(w["w"] for w in p.words[c["i0"]:c["i1"] + 1][:7])
        c["caption"] = " ".join(w["w"] for w in p.words[c["i0"]:c["i1"] + 1])[:150]
    p.clips.append(c)
    p.save()
    return new_job("render", slug, lambda j: render_ids(slug, [c["id"]]), f"New clip: {c['title']}")


@app.delete("/api/projects/{slug}/clips/{cid}")
def delete_clip(slug: str, cid: int):
    p = editable(slug)
    try:
        c = p.clip(cid)
    except StopIteration:
        raise HTTPException(404, "no such clip")
    if c.get("file"):
        for ext in (".mp4", ".jpg", ".txt", ".srt"):
            (p.dir / c["file"]).with_suffix(ext).unlink(missing_ok=True)
    p.data["clips"] = [x for x in p.clips if x["id"] != cid]
    p.save()
    cs.write_gallery(p)
    return {"ok": True}


@app.post("/api/projects/{slug}/render")
async def rerender(slug: str, req: Request):
    body = await req.json()
    editable(slug)
    settings = body.get("settings") or None
    return new_job("render", slug, lambda j: render_ids(slug, body.get("ids"), settings), "Re-render all clips")


@app.post("/api/projects/{slug}/repick")
async def repick(slug: str, req: Request):
    """Ask the AI again (e.g. with a new prompt / count / length) and render the new picks."""
    body = await req.json()
    p = editable(slug)
    s = cs.Settings.from_dict({**p.data.get("settings", {}), **(body.get("settings") or {})})
    video = p.data["video"]

    def run(j):
        q = cs.create_project(video, CLIPS, s, reselect=True)
        cs.render_all(q, only={c["id"] for c in q.clips if not c.get("file")} or None)
    return new_job("create", slug, run, "Find new moments")


@app.delete("/api/projects/{slug}")
def delete_project(slug: str):
    p = editable(slug)
    shutil.rmtree(p.dir)
    return {"ok": True}


@app.get("/files/{slug}/{name}")
def files(slug: str, name: str):
    p = project(slug)
    f = (p.dir / name).resolve()
    if f.parent != p.dir.resolve() or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("CLIPSMITH_PORT", "8765"))
    print(f"clipsmith: http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
