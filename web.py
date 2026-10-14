#!/usr/bin/env -S uv run --project /home/jt/claude/clipsmith python
"""Designer Studio web app: pick or upload a video (or paste a URL), watch it work, then review, edit and
re-render the shorts. Local only (127.0.0.1). One GPU job at a time, in a queue.

    ./serve.sh            # http://127.0.0.1:8765
"""
import json
import os
import queue
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response

import clipsmith as cs
import studio
import users as U
from contextvars import ContextVar
from fastapi.responses import JSONResponse

ME = ContextVar("me", default=None)      # the signed-in user for this request (or the job's owner)

HERE = Path(__file__).resolve().parent
CLIPS = HERE / "clips"
UPLOADS = HERE / "uploads"
FOLDERS_FILE = HERE / "library.json"      # extra library folders added in the app (external drives...)


def my_folders():
    """Extra library folders: the user's own (from setup), or the shared list from before users existed."""
    if uid():
        return U.prefs(uid())["import"].get("folders", [])
    try:
        return json.loads(FOLDERS_FILE.read_text()).get("folders", [])
    except (OSError, ValueError):
        return []


def library_roots():
    extra = [Path(p).expanduser() for p in my_folders()]
    return LIBRARY + [p for p in extra if p not in LIBRARY]


LIBRARY = [Path(p).expanduser() for p in os.environ.get("CLIPSMITH_LIBRARY", "~/Videos").split(":")] + \
          [UPLOADS, HERE / "downloads"]
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}

app = FastAPI(title="Designer Studio")

# Open without signing in: the home screen's profile picker, creating a user, signing in.
OPEN_PATHS = {("GET", "/api/users"), ("POST", "/api/users"), ("POST", "/api/login"), ("GET", "/api/me"),
              ("POST", "/api/logout"), ("GET", "/api/meta")}


@app.middleware("http")
async def signed_in(request: Request, call_next):
    u = U.session_user(request.cookies.get("ds_session"))
    token = ME.set(u)
    try:
        path = request.url.path
        guarded = path.startswith("/api/") or path.startswith("/files/")
        public_photo = request.method == "GET" and path.startswith("/api/users/") and path.endswith("/photo")
        if guarded and not u and U.all_users() and (request.method, path) not in OPEN_PATHS and not public_photo:
            return JSONResponse({"detail": "Please sign in."}, status_code=401)
        return await call_next(request)
    finally:
        ME.reset(token)


def me():
    return ME.get()


def uid():
    u = me()
    return u["id"] if u else None


def is_admin():
    u = me()
    return bool(u and u.get("admin")) or not U.all_users()


def can_see(owner):
    """Projects belong to their owner; ones from before users existed belong to the admin."""
    if not U.all_users():
        return True
    return owner == uid() or (owner is None and is_admin())


def user_settings():
    """Defaults for new work: built-in < brand.json < the user's choices from setup."""
    base = {**cs.Settings().to_dict(), **cs.load_brand(HERE / "brand.json")}
    if uid():
        base.update({k: v for k, v in U.prefs(uid()).get("settings", {}).items() if k in base})
    return base


def suffix():
    """Project folders of non-admin users get a short suffix so two people can import the same video."""
    u = me()
    return "" if not u or u.get("admin") else "-" + u["id"][:4]


def ig_config():
    return (U.user_dir(uid()) / "instagram.json") if uid() else None


def drive_remote():
    return f"gdrive_{uid()}" if uid() else "gdrive"


def profile_path():
    return (U.user_dir(uid()) / "profile.json") if uid() else cs.PROFILE_FILE
from fastapi.staticfiles import StaticFiles  # noqa: E402
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
JOBS = {}            # id -> job dict
Q = queue.Queue()
CURRENT = None       # the job the worker is running
ACTIVE = ("queued", "running", "paused", "stopping")


class JobStopped(BaseException):
    """Raised at a checkpoint when the user stops the job (BaseException so no `except Exception` swallows it)."""


def children(pid):
    """All descendant process ids of pid (ffmpeg, the dubbing voice...)."""
    kids = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            if ppid == pid:
                kids.append(int(d))
    return [x for k in kids for x in [k] + children(k)]


def signal_children(sig):
    for pid in children(os.getpid()):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass


def job_check():
    j = CURRENT
    if j is None or threading.get_ident() != j.get("_tid"):
        return
    if j["_cancel"]:
        raise JobStopped()
    if j["_pause"]:
        j["status"] = "paused"
        while j["_pause"] and not j["_cancel"]:
            time.sleep(0.3)
        if j["_cancel"]:
            raise JobStopped()
        j["status"] = "running"


cs._checks.append(job_check)


# ---------------------------------------------------------------- job queue (one GPU job at a time)

def new_job(kind, project, fn, label, ids=None):
    j = {"id": uuid.uuid4().hex[:8], "kind": kind, "project": project, "label": label, "status": "queued", "ids": ids,
         "progress": 0.0, "log": [], "error": None, "result": None, "created": time.time(), "fn": fn,
         "_pause": False, "_cancel": False, "user": uid(), "uname": (me() or {}).get("name", "")}
    JOBS[j["id"]] = j
    U.log(me(), "queued", label)
    Q.put(j)
    return public(j)


def public(j):
    return {k: v for k, v in j.items() if k != "fn" and not k.startswith("_")} | {"log": j["log"][-40:]}


def busy(slug):
    return any(j["project"] == slug and j["status"] in ACTIVE for j in JOBS.values())


def creating(slug):
    """An import or "find new moments" is replacing this project's clips: no edits until it's done."""
    return any(j["project"] == slug and j["kind"] == "create" and j["status"] in ACTIVE for j in JOBS.values())


def clip_states(slug):
    """{clip id: "queued" | "rendering"} from the queue (a whole-project render covers every clip)."""
    out = {}
    for j in sorted(JOBS.values(), key=lambda j: j["created"]):
        if j["project"] != slug or j["kind"] != "render" or j["status"] not in ACTIVE:
            continue
        state = "rendering" if j["status"] in ("running", "paused", "stopping") else "queued"
        for cid in (j["ids"] or ["*"]):
            if out.get(cid) != "rendering":
                out[cid] = state
    return out


def queue_render(slug, ids, label, settings=None):
    """Queue a render; if the same clips are already waiting in the queue, that job will pick up the latest
    edits when it starts, so don't add another."""
    for j in JOBS.values():
        if (j["project"] == slug and j["kind"] == "render" and j["status"] == "queued" and not settings
                and j.get("_settings") is None and j["ids"] and ids and set(ids) <= set(j["ids"])):
            return public(j) | {"merged": True}
    j = new_job("render", slug, lambda j: render_ids(slug, ids, settings), label, ids)
    JOBS[j["id"]]["_settings"] = settings
    return j | {"merged": False}


def worker():
    global CURRENT
    while True:
        j = Q.get()
        if j["_cancel"]:                 # stopped while it was waiting in the queue
            continue
        j["status"], j["started"], j["_tid"] = "running", time.time(), threading.get_ident()
        CURRENT = j
        owner = U.get(j["user"]) if j.get("user") else None
        me_token = ME.set(owner)                       # the job runs as the person who started it
        cs.PROFILE_FILE = (U.user_dir(owner["id"]) / "profile.json") if owner else HERE / "profile.json"
        cs.OWNER = owner["id"] if owner else None

        def hook(msg, progress, j=j):
            if msg is not None:
                j["log"].append(msg)
            if progress is not None:
                j["progress"] = progress
        cs._hooks.append(hook)
        try:
            j["result"] = j["fn"](j)
            j["status"], j["progress"] = "done", 1.0
        except JobStopped:
            j["status"] = "stopped"
            j["log"].append("Stopped. Clips finished before the stop are kept.")
        except SystemExit as e:          # clipsmith's friendly errors (GPU full, no speech, ...)
            j["status"], j["error"] = ("stopped", None) if j["_cancel"] else ("error", str(e))
        except Exception as e:  # noqa: BLE001
            if j["_cancel"]:             # a helper we killed on stop made this step fail
                j["status"] = "stopped"
                j["log"].append("Stopped. Clips finished before the stop are kept.")
            else:
                j["status"], j["error"] = "error", f"{e}"
                j["log"].append(traceback.format_exc()[-2000:])
        finally:
            cs._hooks.remove(hook)
            j["finished"] = time.time()
            CURRENT = None
            U.log(owner, {"done": "finished", "error": "failed", "stopped": "stopped"}.get(j["status"], j["status"]),
                  j["label"] + (f" — {j['error']}" if j.get("error") else ""))
            ME.reset(me_token)


threading.Thread(target=worker, daemon=True).start()


@app.delete("/api/jobs/finished")
def clear_finished():
    """Remove finished, stopped and failed jobs from the list."""
    for jid in [k for k, j in JOBS.items() if j["status"] not in ACTIVE]:
        del JOBS[jid]
    return {"ok": True}


@app.post("/api/jobs/{jid}/{action}")
def job_action(jid: str, action: str):
    """pause / resume / stop a job. Pause freezes running helpers (ffmpeg, the dubbing voice) right away and
    the job itself at its next step; stop ends it (or drops it from the queue) and keeps finished clips."""
    import signal
    j = JOBS.get(jid)
    if j is None:
        raise HTTPException(404, "no such job")
    st = j["status"]
    if action == "pause" and st == "running":
        j["_pause"] = True
        j["status"] = "paused"
        if j is CURRENT:
            signal_children(signal.SIGSTOP)
    elif action == "resume" and st == "paused":
        j["_pause"] = False
        j["status"] = "running"
        if j is CURRENT:
            signal_children(signal.SIGCONT)
    elif action == "stop" and st in ACTIVE:
        j["_cancel"], j["_pause"] = True, False
        if st == "queued":
            j["status"], j["finished"] = "stopped", time.time()
            j["log"].append("Removed from the queue.")
        else:
            j["status"] = "stopping"
            if j is CURRENT:
                signal_children(signal.SIGCONT)
                signal_children(signal.SIGTERM)
    else:
        raise HTTPException(409, f"can't {action} a job that is {st}")
    return public(j)


# ---------------------------------------------------------------- helpers

def project(slug):
    p = cs.Project(CLIPS / slug)
    if not p.file.exists() or "/" in slug or slug.startswith(".") or not can_see(p.data.get("owner")):
        raise HTTPException(404, "no such project")
    return p


def own(p):
    """Mark a new project as the signed-in user's."""
    if uid() and p.data.get("owner") != uid():
        p.data["owner"] = uid()
        p.save()


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
    import dub
    return {"defaults": user_settings(), "prefs": U.prefs(uid()) if uid() else U.DEFAULT_PREFS,
            "captions": list(cs.CAPTION_STYLES) + ["none"],
            "dub_langs": dub.DUB_LANGS,
            "aspects": list(cs.ASPECTS), "layouts": ["auto", "track", "split", "fit", "center"]}


@app.get("/api/state")
def state():
    projects = []
    for f in sorted(CLIPS.glob("*/project.json"), key=lambda f: -f.stat().st_mtime):
        try:
            p = cs.Project(f.parent)
            if can_see(p.data.get("owner")):
                projects.append(summary(p))
        except Exception:  # noqa: BLE001
            pass
    jobs = sorted([j for j in JOBS.values() if not U.all_users() or j.get("user") == uid()],
                  key=lambda j: -j["created"])[:20]
    return {"projects": projects, "jobs": [public(j) for j in jobs]}


@app.get("/api/library")
def library():
    out = []
    for root in library_roots():
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
    sources = [x.strip() for x in (body.get("sources") or [body.get("source") or ""]) if x and x.strip()]
    if not sources:
        raise HTTPException(400, "pick a video or paste a URL")
    s = cs.Settings.from_dict({**user_settings(), **(body.get("settings") or {})})
    jobs = []
    for source in sources:
        def run(j, source=source, sfx=suffix()):
            p = cs.create_project(source, CLIPS, s, reselect=bool(body.get("reselect")), suffix=sfx)
            own(p)
            j["project"] = p.dir.name
            cs.render_all(p)
        label = Path(source).name if "://" not in source else source
        jobs.append(new_job("create", cs.slug(Path(source).stem) + suffix() if "://" not in source else None, run,
                            f"New clips: {label}"))
    return jobs[0] if len(jobs) == 1 else {"jobs": jobs}


@app.get("/api/projects/{slug}")
def get_project(slug: str):
    p = project(slug)
    return {**p.data, "slug": slug, "busy": busy(slug), "creating": creating(slug), "queue": clip_states(slug)}


@app.get("/api/projects/{slug}/transcript")
def transcript(slug: str):
    p = project(slug)
    return [{"i": i, "s": x["s"], "e": x["e"], "text": x["text"]} for i, x in enumerate(cs.sentences_from_words(p.words))]


def editable(slug):
    if creating(slug):
        raise HTTPException(409, "this project is still being analyzed; edits open when that finishes")
    return project(slug)


@app.patch("/api/projects/{slug}/clips/{cid}")
async def edit_clip(slug: str, cid: int, req: Request):
    body = await req.json()
    p = editable(slug)
    try:
        c = p.clip(cid)
    except StopIteration:
        raise HTTPException(404, "no such clip")
    for k in ("title", "caption", "hashtags", "keywords", "platforms", "topic", "metrics", "broll"):
        if k in body:
            c[k] = body[k]
    if "platforms" in body:     # the Instagram caption is the clip's main caption
        ig = body["platforms"].get("instagram", {})
        c["caption"], c["hashtags"] = ig.get("caption", c.get("caption", "")), ig.get("hashtags", c.get("hashtags", []))
    if "start" in body or "end" in body:
        try:
            p.set_range(c, body.get("start", c["start"]), body.get("end", c["end"]))
        except ValueError as e:
            raise HTTPException(400, str(e))
    if "overrides" in body:
        c["overrides"] = {k: v for k, v in body["overrides"].items() if v not in (None, "")}
    p.save()
    if body.get("render", True):
        return queue_render(slug, [cid], f"Re-render: {c.get('title', '')}")
    return {"ok": True}


def render_ids(slug, ids, settings=None):
    p = project(slug)
    if settings:
        p.data["settings"] = {**p.data.get("settings", {}), **settings}
        p.save()
    cs.render_all(p, only=set(ids) if ids else None, start=0.0)


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
    return queue_render(slug, [c["id"]], f"New clip: {c['title']}")


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
    U.log(me(), "deleted clip", c.get("title", ""))
    cs.write_gallery(p)
    return {"ok": True}


@app.post("/api/projects/{slug}/render")
async def rerender(slug: str, req: Request):
    body = await req.json()
    editable(slug)
    settings = body.get("settings") or None
    return queue_render(slug, body.get("ids"), "Re-render all clips" if not body.get("ids") else "Re-render clips", settings)


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


@app.post("/api/projects/{slug}/stitch")
async def stitch(slug: str, req: Request):
    """Join clips into one video: ids (default all rendered), order source|score|custom, transition cut|fade."""
    body = await req.json()
    p = editable(slug)
    ids = body.get("ids") or None
    order = body.get("order", "source")
    if order not in ("source", "score", "custom"):
        raise HTTPException(400, "order must be source, score or custom")
    if len([c for c in p.clips if c.get("file") and (not ids or c["id"] in ids)]) < 2:
        raise HTTPException(400, "pick at least two rendered clips")
    return new_job("stitch", slug, lambda j: cs.stitch(project(slug), ids, order, body.get("transition", "cut")),
                   f"Stitch {len(ids) if ids else 'all'} clips")


@app.delete("/api/projects/{slug}/compilations/{n}")
def delete_compilation(slug: str, n: int):
    p = editable(slug)
    for rec in [r for r in p.data.get("compilations", []) if r["n"] == n]:
        for ext in (".mp4", ".jpg"):
            (p.dir / rec["file"]).with_suffix(ext).unlink(missing_ok=True)
    p.data["compilations"] = [r for r in p.data.get("compilations", []) if r["n"] != n]
    p.save()
    return {"ok": True}


@app.get("/api/projects/{slug}/sheet")
def sheet(slug: str, start: str = "", per_day: int = 1, platforms: str = "", fmt: str = "json"):
    """Clip sheet, optionally with a posting schedule (start=YYYY-MM-DD, platforms=instagram,tiktok,youtube)."""
    p = project(slug)
    plats = [x for x in platforms.split(",") if x in ("instagram", "tiktok", "youtube")]
    plan = cs.schedule(p, start or None, per_day, plats, CLIPS) if plats else None
    rows = cs.clip_sheet(p, plan)
    if fmt == "csv":
        out = p.dir / "clip_sheet.csv"
        cs.write_sheet_csv(out, rows)
        return FileResponse(out, media_type="text/csv", filename=f"{slug}_clip_sheet.csv")
    return rows


def read_profile():
    try:
        return {**cs.PROFILE_FIELDS, **json.loads(profile_path().read_text())}
    except (OSError, ValueError):
        return dict(cs.PROFILE_FIELDS)


@app.get("/api/profile")
def get_profile():
    return read_profile()


@app.put("/api/profile")
async def put_profile(req: Request):
    body = await req.json()
    data = {k: str(body.get(k, "")).strip() for k in cs.PROFILE_FIELDS}
    profile_path().write_text(json.dumps(data, indent=1, ensure_ascii=False))
    U.log(me(), "updated creator profile")
    return read_profile()


@app.get("/api/insights")
def get_insights():
    return cs.insights(CLIPS, owner=uid())


@app.post("/api/insights/ai")
def ai_insights():
    s = cs.Settings.from_dict(user_settings())
    return new_job("insights", None, lambda j: cs.insights(CLIPS, s, owner=cs.OWNER)["ai"], "AI insights")


# ---------------------------------------------------------------- YouTube & Instagram tools

def studio_job(slug, label, fn):
    def run(j):
        p = project(slug)
        return fn(p, p.settings)
    return new_job("studio", slug, run, label)


@app.post("/api/projects/{slug}/youtube")
async def youtube(slug: str, req: Request):
    body = await req.json()
    project(slug)
    if body.get("action") == "refine":
        msg = (body.get("message") or "").strip()
        if not msg:
            raise HTTPException(400, "say how to change the titles")
        return studio_job(slug, "YouTube: refine titles", lambda p, s: studio.youtube_refine(p, s, msg))
    return studio_job(slug, "YouTube: titles, description, tags", studio.youtube_package)


@app.put("/api/projects/{slug}/youtube")
async def youtube_save(slug: str, req: Request):
    body = await req.json()
    p = editable(slug)
    yt = p.data.setdefault("youtube", {})
    for k in ("titles", "chosen", "description", "chapters", "tags", "pinned_comment"):
        if k in body:
            yt[k] = body[k]
    p.save()
    return yt


@app.post("/api/projects/{slug}/frames")
async def frames(slug: str, req: Request):
    """Good frames for a thumbnail (whole video) or a Reels cover (one clip)."""
    body = await req.json()
    p = project(slug)
    cid = body.get("clip")

    def run(p, s):
        if cid is None:
            res = studio.frame_candidates(p, tag="cand")
            p.data.setdefault("youtube", {})["candidates"] = res
        else:
            c = p.clip(cid)
            res = studio.frame_candidates(p, c["start"], c["end"], n=6, tag=f"cand{cid}")
            c.setdefault("ig", {})["candidates"] = res
        p.save()
        return res
    return studio_job(slug, "Finding the best frames", run)


@app.post("/api/projects/{slug}/thumbnail")
async def thumbnail(slug: str, req: Request):
    body = await req.json()
    project(slug)
    t, text, style = float(body.get("t", 0)), (body.get("text") or "").strip(), body.get("style", "right")
    if not text:
        raise HTTPException(400, "type the thumbnail text")
    return studio_job(slug, "Making a thumbnail", lambda p, s: studio.make_thumbnail(p, t, text, style))


@app.delete("/api/projects/{slug}/thumbnail/{n}")
def delete_thumbnail(slug: str, n: int):
    p = editable(slug)
    yt = p.data.setdefault("youtube", {})
    th = yt.get("thumbs", [])
    if 0 <= n < len(th):
        (p.dir / th[n]["file"]).unlink(missing_ok=True)
        del th[n]
        p.save()
    return {"ok": True}


@app.post("/api/projects/{slug}/clips/{cid}/instagram")
async def instagram(slug: str, cid: int, req: Request):
    body = await req.json()
    project(slug)
    if body.get("action") == "refine":
        msg = (body.get("message") or "").strip()
        if not msg:
            raise HTTPException(400, "say how to change the captions")
        return studio_job(slug, "Instagram: refine captions", lambda p, s: studio.instagram_refine(p, s, p.clip(cid), msg))
    return studio_job(slug, "Instagram: captions, hashtags, alt text",
                      lambda p, s: studio.instagram_package(p, s, p.clip(cid)))


@app.post("/api/projects/{slug}/clips/{cid}/cover")
async def cover(slug: str, cid: int, req: Request):
    body = await req.json()
    project(slug)
    t, text, style = float(body.get("t", 0)), (body.get("text") or "").strip(), body.get("style", "stroke")
    return studio_job(slug, "Making a Reels cover", lambda p, s: studio.make_cover(p, p.clip(cid), t, text, style))


@app.post("/api/projects/{slug}/clips/{cid}/dub")
async def dub_clip(slug: str, cid: int, req: Request):
    import dub
    body = await req.json()
    lang = body.get("lang", "en")
    if lang not in dub.DUB_LANGS:
        raise HTTPException(400, "unsupported language")
    p = project(slug)
    c = p.clip(cid)
    if not c.get("file"):
        raise HTTPException(400, "render the clip first")
    return studio_job(slug, f"Dub into {dub.DUB_LANGS[lang]}: {c.get('title', '')}", lambda p, s: dub.dub_clip(p, p.clip(cid), lang))


@app.delete("/api/projects/{slug}/clips/{cid}/dub/{lang}")
def delete_dub(slug: str, cid: int, lang: str):
    p = editable(slug)
    c = p.clip(cid)
    d = c.get("dubs", {}).pop(lang, None)
    if d:
        for ext in (".mp4", ".jpg", ".srt"):
            (p.dir / d["file"]).with_suffix(ext).unlink(missing_ok=True)
    p.save()
    return {"ok": True}


# ---------------------------------------------------------------- import sources: folders, Google Drive, other links

@app.get("/api/sources")
def sources():
    import shutil as _sh
    remotes = []
    import drive_connect as dc
    exe = dc.rclone()
    if exe:
        try:
            out = __import__("subprocess").run([exe, "listremotes", "--long"], capture_output=True, text=True, timeout=10).stdout
            for line in out.splitlines():
                name, _, kind = line.partition(":")
                if kind.strip() in ("drive", "onedrive", "dropbox", "box", "pcloud", "s3") and \
                        (not U.all_users() or name.strip() == drive_remote()):   # each user sees their own drive
                    remotes.append({"name": name.strip(), "type": kind.strip()})
        except Exception:  # noqa: BLE001
            pass
    folders = my_folders()
    return {"rclone": bool(exe), "remotes": remotes, "folders": folders,
            "builtin": [str(p).replace(str(Path.home()), "~") for p in LIBRARY]}


@app.put("/api/sources/folders")
async def set_folders(req: Request):
    body = await req.json()
    folders = []
    for f in body.get("folders", []):
        p = Path(str(f)).expanduser()
        if not p.is_dir():
            raise HTTPException(400, f"not a folder: {f}")
        folders.append(str(p))
    if uid():
        U.save_prefs(uid(), {"import": {"folders": folders}})
    else:
        FOLDERS_FILE.write_text(json.dumps({"folders": folders}, indent=1))
    U.log(me(), "changed library folders", ", ".join(folders))
    return sources()


VIDEO_RE = r"\.(mp4|mov|mkv|webm|m4v|avi)$"


@app.post("/api/links/list")
async def links_list(req: Request):
    """What's behind a share link: one video, or the videos in a shared folder (Google Drive, Dropbox...)."""
    import re as _re
    import yt_dlp
    import gdrive
    url = ((await req.json()).get("url") or "").strip()
    if not _re.match(r"https?://", url):
        raise HTTPException(400, "paste a link starting with https://")
    if gdrive.is_drive(url):
        try:
            kind, fid = gdrive.parse(url)
            if kind == "folder":
                f = gdrive.list_folder(fid)
                vids = [i for i in f["items"] if i["video"]]
                if not vids:
                    raise ValueError(f"No videos in “{f['title']}” ({len(f['items'])} other files). Subfolders aren't opened.")
                return {"kind": "folder", "title": f["title"],
                        "items": [{"title": i["name"], "url": f"https://drive.google.com/file/d/{i['id']}/view"} for i in vids]}
            info = gdrive.file_info(fid)
            if not info["type"].startswith("video/") and not gdrive.VIDEO_RE.search(info["name"]):
                raise ValueError(f"“{info['name']}” isn't a video.")
            return {"kind": "file", "title": info["name"], "items": [{"title": info["name"], "url": url,
                                                                      "size": info["size"]}]}
        except ValueError as e:
            raise HTTPException(400, str(e))
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True, "extract_flat": "in_playlist", "noplaylist": False}) as y:
            info = y.extract_info(url, download=False)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "drive.google" in url and any(k in msg.lower() for k in ("not public", "403", "404", "sign in", "unable to download")):
            msg = ("Couldn't open that Drive link. Check it's right and shared: in Drive, Share → General access → "
                   "Anyone with the link.")
        raise HTTPException(400, msg.replace("ERROR: ", "")[:300])
    if info.get("_type") == "playlist" or info.get("entries"):
        items = [{"title": e.get("title") or e.get("id"), "url": e.get("url") or e.get("webpage_url"), "duration": e.get("duration")}
                 for e in (info.get("entries") or []) if e]
        items = [i for i in items if i["url"]]
        return {"kind": "folder", "title": info.get("title", ""), "items": items}
    return {"kind": "file", "title": info.get("title", ""), "items": [{"title": info.get("title") or "video", "url": url,
                                                                       "duration": info.get("duration")}]}


@app.post("/api/drive/connect")
def drive_connect():
    import drive_connect as dc
    U.log(me(), "started connecting Google Drive")
    return dc.start(drive_remote())


@app.get("/api/drive/status")
def drive_status():
    import drive_connect as dc
    st = dc.status(drive_remote())
    if st["status"] == "connected" and uid() and not U.prefs(uid())["import"].get("drive_logged"):
        U.save_prefs(uid(), {"import": {"drive_logged": True}})
        U.log(me(), "connected Google Drive")
    return st


@app.delete("/api/drive/connect")
def drive_disconnect():
    import drive_connect as dc
    U.log(me(), "disconnected Google Drive")
    if uid():
        U.save_prefs(uid(), {"import": {"drive_logged": False}})
    return dc.disconnect(drive_remote())


@app.get("/api/drive/browse")
def drive_browse(remote: str, path: str = ""):
    """Folders and videos in a cloud drive connected with rclone."""
    import re as _re
    import subprocess as sp
    if not _re.match(r"^[\w.-]+$", remote) or ".." in path or (U.all_users() and remote != drive_remote()):
        raise HTTPException(400, "bad path")
    import drive_connect as dc
    r = sp.run([dc.rclone() or "rclone", "lsjson", f"{remote}:{path}", "--no-modtime", "--fast-list"], capture_output=True, text=True, timeout=60)
    if r.returncode:
        raise HTTPException(400, r.stderr.strip()[-200:] or "couldn't list the drive")
    items = json.loads(r.stdout or "[]")
    out = [{"name": i["Name"], "dir": i["IsDir"], "size": i.get("Size", 0), "path": (path.rstrip("/") + "/" if path else "") + i["Name"]}
           for i in items if i["IsDir"] or _re.search(VIDEO_RE, i["Name"], _re.I)]
    return sorted(out, key=lambda x: (not x["dir"], x["name"].lower()))


# ---------------------------------------------------------------- users, sign-in, preferences, activity

FAILS = {}               # user id -> [failed passcodes, locked until]


def _signed_in_response(u, body):
    r = JSONResponse(body)
    r.set_cookie("ds_session", U.new_session(u["id"]), max_age=U.SESSION_DAYS * 86400, httponly=True, samesite="lax")
    return r


@app.get("/api/users")
def users_list():
    return [U.public(u) for u in U.all_users()]


@app.post("/api/users")
async def users_create(req: Request):
    """The new-user setup's first step: name (+ optional passcode). Signs the new user in."""
    body = await req.json()
    try:
        u = U.create(body.get("name", ""), (body.get("passcode") or "").strip())
    except ValueError as e:
        raise HTTPException(400, str(e))
    U.log(u, "created user" + (" (admin)" if u["admin"] else ""))
    return _signed_in_response(u, {"user": U.public(u)})


@app.post("/api/login")
async def login(req: Request):
    body = await req.json()
    u = U.get(body.get("id", ""))
    if not u:
        raise HTTPException(404, "No such user.")
    n, until = FAILS.get(u["id"], [0, 0])
    if time.time() < until:
        raise HTTPException(429, f"Too many wrong passcodes. Try again in {int(until - time.time())} seconds.")
    if not U.check_passcode(u, (body.get("passcode") or "").strip()):
        n += 1
        FAILS[u["id"]] = [n, time.time() + 60 if n >= 5 else 0]
        U.log(u, "wrong passcode")
        raise HTTPException(403, "That passcode isn't right.")
    FAILS.pop(u["id"], None)
    U.log(u, "signed in")
    return _signed_in_response(u, {"user": U.public(u)})


@app.post("/api/logout")
def logout(request: Request):
    if me():
        U.log(me(), "signed out")
    U.end_session(request.cookies.get("ds_session"))
    r = JSONResponse({"ok": True})
    r.delete_cookie("ds_session")
    return r


@app.get("/api/me")
def whoami():
    u = me()
    return {"user": U.public(u) if u else None, "users_exist": bool(U.all_users()),
            "prefs": U.prefs(u["id"]) if u else None, "profile": read_profile() if u else None}


@app.put("/api/me")
async def update_me(req: Request):
    """Save the setup / settings: name, passcode, preferences (settings, socials, import, images), creator profile."""
    body = await req.json()
    u = me()
    if not u:
        raise HTTPException(401, "Please sign in.")
    changes = []
    try:
        fields = {k: body[k] for k in ("name", "passcode") if k in body}
        if fields:
            U.update(u["id"], **fields)
            changes += [("passcode" if k == "passcode" else k) for k in fields]
    except ValueError as e:
        raise HTTPException(400, str(e))
    if body.get("prefs"):
        prefs = body["prefs"]
        if "settings" in prefs:
            allowed = set(cs.Settings().to_dict())
            prefs["settings"] = {k: v for k, v in prefs["settings"].items() if k in allowed}
        U.save_prefs(u["id"], prefs)
        changes += list(prefs)
    if body.get("profile") is not None:
        data = {k: str(body["profile"].get(k, "")).strip() for k in cs.PROFILE_FIELDS}
        profile_path().write_text(json.dumps(data, indent=1, ensure_ascii=False))
        changes.append("creator profile")
    if changes:
        U.log(u, "updated settings", ", ".join(changes))
    return whoami()


@app.post("/api/me/upload/{kind}")
async def me_upload(kind: str, file: UploadFile):
    """Profile photo, brand logo (PNG) or presenter photo for AI shorts."""
    u = me()
    if not u or kind not in ("photo", "logo", "presenter"):
        raise HTTPException(400, "bad upload")
    ext = Path(file.filename or "x.png").suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".webp"):
        raise HTTPException(400, "Use a PNG or JPG image.")
    data = await file.read()
    if len(data) > 20 << 20:
        raise HTTPException(400, "That image is over 20 MB.")
    d = U.user_dir(u["id"])
    for old in d.glob(f"{kind}.*"):
        old.unlink()
    dest = d / f"{kind}{ext}"
    dest.write_bytes(data)
    if kind == "photo":
        U.update(u["id"], photo=True)
    elif kind == "logo":
        U.save_prefs(u["id"], {"settings": {"logo": str(dest)}})
    else:
        U.save_prefs(u["id"], {"images": {"presenter": str(dest)}})
    U.log(u, f"uploaded {kind}")
    return whoami()


@app.get("/api/users/{user_id}/photo")
def user_photo(user_id: str):
    if not user_id.isalnum():
        raise HTTPException(404)
    f = next((p for p in (U.DIR / user_id).glob("photo.*")), None) if (U.DIR / user_id).is_dir() else None
    if not f:
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/api/me/image/{kind}")
def my_image(kind: str):
    u = me()
    f = next((p for p in U.user_dir(u["id"]).glob(f"{kind}.*")), None) if u and kind in ("logo", "presenter") else None
    if not f:
        raise HTTPException(404)
    return FileResponse(f)


@app.delete("/api/users/{user_id}")
def users_delete(user_id: str):
    if not is_admin() or not me():
        raise HTTPException(403, "Only the admin can remove users.")
    if user_id == uid():
        raise HTTPException(400, "You can't remove yourself.")
    victim = U.get(user_id)
    if not victim:
        raise HTTPException(404)
    U.delete(user_id)
    U.log(me(), "removed user", victim["name"])
    return users_list()


@app.get("/api/activity")
def activity(everyone: bool = False):
    """Your activity; the admin can see everyone's."""
    if everyone and not is_admin():
        raise HTTPException(403, "Only the admin can see everyone's activity.")
    return U.activity(None if everyone else uid())


# ---------------------------------------------------------------- Clip Generator (guided path)

def gen_job(slug, label, fn, kind="gen"):
    def run(j):
        p = project(slug) if slug else None
        return fn(p)
    return new_job(kind, slug, run, label)


@app.post("/api/gen/analyze")
async def gen_analyze(req: Request):
    """Step 1: transcribe, map and find the moments (no rendering yet)."""
    body = await req.json()
    source = (body.get("source") or "").strip()
    if not source:
        raise HTTPException(400, "pick a video first")
    s = cs.Settings.from_dict({**user_settings(), "count": int(body.get("count") or user_settings().get("count") or 5),
                               "prompt": (body.get("prompt") or "").strip(), "max": float(body.get("max") or 60)})

    def run(j, sfx=suffix()):
        p = cs.create_project(source, CLIPS, s, reselect=True, suffix=sfx)
        own(p)
        j["project"] = p.dir.name
        return {"slug": p.dir.name}
    label = Path(source).name if "://" not in source else source
    return new_job("create", cs.slug(Path(source).stem) + suffix() if "://" not in source else None, run, f"Analyze: {label}")


@app.post("/api/gen/broll")
async def gen_broll(req: Request):
    import generator
    body = await req.json()
    slug, cid = body["slug"], int(body["clip"])
    project(slug)
    if body.get("redraw") is not None:
        k = int(body["redraw"])
        return gen_job(slug, "B-roll: new image", lambda p: generator.broll_redraw(p, p.clip(cid), k, body.get("prompt")))
    return gen_job(slug, "B-roll: choosing and drawing",
                   lambda p: generator.broll_from_ideas(p, p.clip(cid)) if p.clip(cid).get("broll_ideas")
                   else generator.broll_suggest(p, p.clip(cid), int(body.get("count") or 3)))


@app.post("/api/gen/look")
async def gen_look(req: Request):
    import generator
    body = await req.json()
    slug, cid = body["slug"], int(body["clip"])
    project(slug)
    return gen_job(slug, "Choosing a look", lambda p: generator.pick_look(p, p.clip(cid)))


@app.post("/api/gen/finish")
async def gen_finish(req: Request):
    """Last step: apply the chosen style to the chosen clips and queue their renders (and dubs)."""
    import dub
    body = await req.json()
    slug, ids = body["slug"], [int(i) for i in body.get("ids") or []]
    p = editable(slug)
    style = {k: v for k, v in (body.get("style") or {}).items() if k in ("captions", "look", "aspect", "layout") and v}
    for c in p.clips:
        if c["id"] in ids:
            c["overrides"] = {**c.get("overrides", {}), **style}
    p.save()
    jobs = [queue_render(slug, ids, "Rendering your short" + ("s" if len(ids) > 1 else ""))]
    lang = body.get("dub")
    if lang and lang in dub.DUB_LANGS:
        for cid in ids:
            jobs.append(studio_job(slug, f"Dub into {dub.DUB_LANGS[lang]}: {p.clip(cid).get('title', '')}",
                                   lambda p, s, cid=cid: dub.dub_clip(p, p.clip(cid), lang)))
    for cid in ids:     # the Instagram kit (captions, hashtags, alt text) for the export screen
        if not p.clip(cid).get("ig"):
            jobs.append(studio_job(slug, f"Instagram: {p.clip(cid).get('title', '')}",
                                   lambda p, s, cid=cid: studio.instagram_package(p, s, p.clip(cid))))
    return {"jobs": jobs}


# No video: AI Shorts
@app.post("/api/shorts")
async def shorts_new(req: Request):
    import generator
    body = await req.json()
    if not (body.get("description") or body.get("url")):
        raise HTTPException(400, "describe the video or paste a link")
    sh = generator.Short()
    sh.data.update({k: body.get(k) for k in ("description", "url", "language", "seconds", "actor", "voice", "audience")})
    sh.data["owner"] = uid()
    pres = U.prefs(uid())["images"].get("presenter") if uid() else ""
    if pres and Path(pres).exists():                 # the presenter photo chosen in setup is offered first
        sh.dir.mkdir(parents=True, exist_ok=True)
        dest = sh.dir / f"presenter{Path(pres).suffix}"
        __import__("shutil").copy(pres, dest)
        sh.data["portraits"], sh.data["portrait"] = [str(dest)], str(dest)
    sh.save()
    j = new_job("gen", None, lambda j: generator.write_script(generator.Short(sh.id)), "Writing your script")
    return {"id": sh.id, "job": j}


def short(sid):
    import generator
    if not sid.isalnum():
        raise HTTPException(404)
    sh = generator.Short(sid)
    if not sh.file.exists() or not can_see(sh.data.get("owner")):
        raise HTTPException(404, "no such short")
    return sh


@app.get("/api/shorts/{sid}")
def shorts_get(sid: str):
    sh = short(sid)
    d = dict(sh.data)
    for k in ("portraits",):
        d[k] = [Path(x).name for x in d.get(k, [])]
    for k in ("portrait", "voice_wav", "video"):
        if d.get(k):
            d[k] = Path(d[k]).name
    return d


@app.put("/api/shorts/{sid}")
async def shorts_put(sid: str, req: Request):
    body = await req.json()
    sh = short(sid)
    for k in ("script", "actor", "voice", "language", "seconds"):
        if k in body:
            sh.data[k] = body[k]
    if body.get("portrait"):
        f = sh.dir / Path(body["portrait"]).name
        if f.exists():
            sh.data["portrait"] = str(f)
    sh.save()
    return shorts_get(sid)


@app.post("/api/shorts/{sid}/{step}")
async def shorts_step(sid: str, step: str, req: Request):
    import generator
    sh = short(sid)
    if step == "portraits":
        return new_job("gen", None, lambda j: generator.portraits(generator.Short(sid)), "Creating your AI actor")
    if step == "voice":
        return new_job("gen", None, lambda j: generator.record_voice(generator.Short(sid)), "Recording the voice")
    if step == "video":
        base, sfx = user_settings(), suffix()

        def run(j):
            s2 = generator.Short(sid)
            generator.make_video(s2)
            slug = generator.make_project(s2, CLIPS, base, sfx)
            j["project"] = slug
            return {"slug": slug}
        return new_job("gen", None, run, "Animating your AI actor")
    if step == "photo":                                # your own photo instead of an AI actor
        form = await req.form()
        up = form["file"]
        sh.dir.mkdir(parents=True, exist_ok=True)
        dest = sh.dir / f"upload_{uuid.uuid4().hex[:6]}{Path(up.filename or 'photo.png').suffix or '.png'}"
        dest.write_bytes(await up.read())
        sh.data.setdefault("portraits", []).insert(0, str(dest))
        sh.data["portrait"] = str(dest)
        sh.save()
        return shorts_get(sid)
    raise HTTPException(404)


@app.get("/api/shorts/{sid}/file/{name}")
def shorts_file(sid: str, name: str):
    sh = short(sid)
    f = (sh.dir / name).resolve()
    if f.parent != sh.dir.resolve() or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


# ---------------------------------------------------------------- Instagram posting

@app.get("/api/instagram")
def ig_status():
    import instagram
    return instagram.status(ig_config())


@app.post("/api/instagram/connect")
async def ig_connect(req: Request):
    import instagram
    body = await req.json()
    if not (body.get("token") or "").strip():
        raise HTTPException(400, "paste your access token")
    try:
        r = instagram.connect(body["token"], body.get("user_id", ""), config=ig_config())
        U.log(me(), "connected Instagram", "@" + r.get("username", ""))
        return r
    except SystemExit as e:
        raise HTTPException(400, str(e))


@app.delete("/api/instagram")
def ig_disconnect():
    import instagram
    U.log(me(), "disconnected Instagram")
    return instagram.disconnect(ig_config())


@app.post("/api/projects/{slug}/clips/{cid}/post")
async def ig_post(slug: str, cid: int, req: Request):
    """Post a rendered clip (or one of its dubs) to Instagram as a Reel. The UI asks you to confirm first."""
    import instagram
    body = await req.json()
    p = project(slug)
    c = p.clip(cid)
    lang = body.get("dub")
    f = (c.get("dubs", {}).get(lang) or {}).get("file") if lang else c.get("file")
    if not f:
        raise HTTPException(400, "render the clip first")
    caption = (body.get("caption") or "").strip()
    cover_ms = int(float(body.get("cover_ms") or 1200))
    cfg = ig_config()

    def run(p, s):
        r = instagram.post_reel(p.dir / f, caption, cover_ms, config=cfg)
        U.log(U.get(p.data.get("owner")) if p.data.get("owner") else None, "posted to Instagram",
              f"{c.get('title', '')} {r.get('permalink', '')}")
        c2 = p.clip(cid)
        c2.setdefault("posts", []).append({**r, "platform": "instagram", "dub": lang or ""})
        c2.setdefault("metrics", []).append({"platform": "instagram", "posted_at": time.strftime("%Y-%m-%dT%H:%M"),
                                             "views": 0, "likes": 0, "comments": 0, "shares": 0, "saves": 0})
        p.save()
        return r
    return studio_job(slug, f"Instagram: posting {c.get('title', '')}", run)


@app.delete("/api/projects/{slug}")
def delete_project(slug: str):
    if busy(slug):
        raise HTTPException(409, "stop or wait for this project's jobs first")
    p = project(slug)
    U.log(me(), "deleted project", Path(p.data.get("video", slug)).name)
    shutil.rmtree(p.dir)
    return {"ok": True}


_media_locks = {}


@app.get("/api/projects/{slug}/media/{name}")
def media(slug: str, name: str):
    """Timeline media, made on first use and cached: proxy.mp4 (small playable copy of the source),
    strip.jpg (filmstrip) and wave.png (waveform). Decoding/encoding on the GPU."""
    if name not in ("proxy.mp4", "strip.jpg", "wave.png"):
        raise HTTPException(404)
    p = project(slug)
    out = p.work / name
    with _media_locks.setdefault((slug, name), threading.Lock()):
        if not out.exists():
            info, src = p.data["info"], p.data["video"]
            dur = info["duration"]
            tmp = out.with_name("tmp_" + name)
            if name == "proxy.mp4":
                scale = "scale=-2:960" if info["h"] > info["w"] else "scale=-2:540"
                cmd = ["ffmpeg", "-v", "error", "-y", "-hwaccel", "cuda", "-i", src, "-vf", scale + ",format=yuv420p",
                       "-c:v", "h264_nvenc", "-cq", "28", "-g", "30", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(tmp)]
            elif name == "strip.jpg":
                n = int(min(240, max(24, dur / 1.5)))
                cmd = ["ffmpeg", "-v", "error", "-y", "-hwaccel", "cuda", "-i", src, "-vf",
                       f"fps={n / dur:.6f},scale=-2:72,tile={n}x1", "-frames:v", "1", "-q:v", "4", str(tmp)]
            else:
                cmd = ["ffmpeg", "-v", "error", "-y", "-i", src, "-filter_complex",
                       "aformat=channel_layouts=mono,showwavespic=s=4000x120:colors=0xb8cdff:scale=sqrt", "-frames:v", "1", str(tmp)]
            r = __import__("subprocess").run(cmd, capture_output=True, text=True)
            if r.returncode or not tmp.exists():
                raise HTTPException(500, f"couldn't make {name}: {r.stderr[-300:]}")
            tmp.replace(out)
    return FileResponse(out)


@app.get("/api/projects/{slug}/frame/{tenths}")
def frame(slug: str, tenths: int):
    """A clean frame (no captions) from the source at tenths/10 s, 480 wide, cached."""
    p = project(slug)
    t = max(0.0, min(tenths / 10, p.data["info"]["duration"] - 0.1))
    out = p.work / "frames" / f"{int(t * 10)}.jpg"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        __import__("subprocess").run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", p.data["video"], "-frames:v", "1",
                                      "-vf", "scale=480:-2", "-q:v", "4", str(out)], check=False)
        if not out.exists():
            raise HTTPException(404)
    return FileResponse(out, headers={"Cache-Control": "max-age=86400"})


@app.get("/files/{slug}/{name:path}")
def files(slug: str, name: str):
    p = project(slug)
    f = (p.dir / name).resolve()
    if f.parent not in (p.dir.resolve(), (p.dir / "studio").resolve()) or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("CLIPSMITH_PORT", "8765"))
    print(f"Designer Studio: http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
