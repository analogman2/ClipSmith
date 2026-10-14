"""Designer Studio publishing tools: YouTube (titles, description with chapters, tags, thumbnails) and
Instagram (caption options, hashtag sets, alt text, first comment, Reels covers). The writing uses the
local Ollama model (GPU only, like everything else); images are composed with Pillow from real frames.
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

import clipsmith as cs

FONT = str(cs.DEFAULT_FONT[0])


# ---------------------------------------------------------------- the writer (local LLM)

def llm_json(s: cs.Settings, system, user, schema, num_ctx=16384, temperature=0.7):
    url = s.ollama.rstrip("/")
    cs.ensure_model_on_gpu(url, s.model, num_ctx)
    raw = cs.ollama_chat(url, s.model, system, user, False, num_ctx, schema, temperature)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise SystemExit(f"The model's answer wasn't valid JSON: {raw[:200]}")


def language_name(p):
    lang = p.data.get("language", "")
    return cs.LANGS.get(lang, lang or "the transcript's language")


def digest(p, limit=9000):
    """What the video is about, compactly: chapters, the AI's clips and the start of the transcript."""
    parts = []
    if p.data.get("map"):
        parts.append("Chapters:\n" + "\n".join(f"- {cs.fmt_ts(c['start'])} {c['title']}: {c['summary']}" for c in p.data["map"]))
    if p.clips:
        parts.append("Best moments:\n" + "\n".join(f"- {c.get('title', '')} ({c.get('topic', '')})" for c in p.clips))
    text = " ".join(w["w"] for w in p.words)
    parts.append("Transcript (start):\n" + text[:limit])
    return "\n\n".join(parts)


def profile_text():
    prof = cs.load_profile()
    lines = [f"- {k}: {v}" for k, v in prof.items() if v]
    return ("Creator profile:\n" + "\n".join(lines) + "\n\n") if lines else ""


# ---------------------------------------------------------------- YouTube

YT_PROMPT = """You are a YouTube strategist. From the video summary, write the publishing package for the \
full (long) video. Everything in {language}, in the creator's voice.
- titles: 10 different titles, max 70 characters, each a different angle (curiosity gap, how-to, bold claim, \
number/list, question, story, contrarian, benefit, mistake to avoid, direct "you"). Clickable but honest.
- description: 2 short paragraphs that hook in the first line (it shows in search), then what viewers learn. \
No timestamps (they are added separately). End with a call to action.
- chapters_heading: the word for "Chapters" in {language}.
- tags: 12-15 search tags.
- pinned_comment: one question that invites replies.
- thumbnail_texts: 5 very short thumbnail texts (2-4 words, punchy, readable at phone size), not the title."""

YT_SCHEMA = {"type": "object", "properties": {
    "titles": {"type": "array", "items": {"type": "string"}}, "description": {"type": "string"},
    "chapters_heading": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}},
    "pinned_comment": {"type": "string"}, "thumbnail_texts": {"type": "array", "items": {"type": "string"}}},
    "required": ["titles", "description", "chapters_heading", "tags", "pinned_comment", "thumbnail_texts"]}

REFINE_SCHEMA = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "string"}},
                                                  "note": {"type": "string"}}, "required": ["items", "note"]}


def youtube_chapters(p, heading="Chapters"):
    """YouTube's rules: first chapter at 0:00, at least 3, each at least 10 seconds."""
    chs = [c for c in p.data.get("map", [])]
    out = []
    for c in chs:
        start = 0.0 if not out else c["start"]
        if out and start - out[-1][0] < 10:
            continue
        out.append((start, c["title"]))
    if len(out) < 3:
        return ""
    return heading + ":\n" + "\n".join(f"{cs.fmt_ts(t)} {title}" for t, title in out)


def youtube_package(p, s):
    cs.log("[youtube] writing titles, description and tags...", 0.2)
    d = llm_json(s, YT_PROMPT.format(language=language_name(p)), profile_text() + digest(p), YT_SCHEMA)
    yt = p.data.setdefault("youtube", {})
    yt.update(titles=[t.strip() for t in d["titles"] if t.strip()][:10], chosen=0, description=d["description"].strip(),
              chapters=youtube_chapters(p, d["chapters_heading"].strip() or "Chapters"),
              tags=[t.strip().lstrip("#") for t in d["tags"] if t.strip()], pinned_comment=d["pinned_comment"].strip(),
              thumb_texts=[t.strip() for t in d["thumbnail_texts"] if t.strip()][:5], chat=[])
    p.save()
    cs.log("[youtube] done", 1.0)
    return yt


def refine(s, what, language, current, chat, message, context):
    system = (f"You rewrite {what} for a creator. Keep the same language ({language}) and count. Apply the "
              "creator's feedback precisely; keep what they liked. Return the full new list in items and a one-line "
              "note (in English) on what you changed.")
    history = "\n".join(f"{m['role']}: {m['content']}" for m in chat[-8:])
    user = (f"{context}\n\nCurrent {what}:\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(current)) +
            (f"\n\nEarlier feedback:\n{history}" if history else "") + f"\n\nNew feedback: {message}")
    return llm_json(s, system, user, REFINE_SCHEMA)


def youtube_refine(p, s, message):
    yt = p.data.setdefault("youtube", {})
    cs.log("[youtube] refining titles...", 0.3)
    d = refine(s, "YouTube titles (max 70 characters each)", language_name(p), yt.get("titles", []), yt.get("chat", []),
               message, digest(p, 3000))
    yt["titles"] = [t.strip() for t in d["items"] if t.strip()][:10] or yt.get("titles", [])
    yt["chosen"] = 0
    yt.setdefault("chat", []).extend([{"role": "you", "content": message}, {"role": "ai", "content": d["note"]}])
    p.save()
    return yt


# ---------------------------------------------------------------- Instagram

IG_PROMPT = """You are an Instagram Reels growth expert. Write the posting kit for this Reel, in {language}, \
in the creator's voice.
- captions: 5 different captions. Each starts with a scroll-stopping first line (it is all people see before \
"more"), then 1-3 short lines of value or story, then a call to action that drives saves, shares or comments. \
Use line breaks; at most 2 emojis; no hashtags inside.
- hashtags_broad: 5 popular hashtags for the topic; hashtags_niche: 8 specific hashtags for this niche and \
audience; hashtags_community: 4 hashtags the creator's community uses (no # sign anywhere).
- alt_text: one sentence describing the video for screen readers.
- first_comment: a comment the creator can post first to start the conversation.
- cover_texts: 4 short cover titles (2-5 words) that work in the profile grid."""

IG_SCHEMA = {"type": "object", "properties": {
    "captions": {"type": "array", "items": {"type": "string"}},
    "hashtags_broad": {"type": "array", "items": {"type": "string"}},
    "hashtags_niche": {"type": "array", "items": {"type": "string"}},
    "hashtags_community": {"type": "array", "items": {"type": "string"}},
    "alt_text": {"type": "string"}, "first_comment": {"type": "string"},
    "cover_texts": {"type": "array", "items": {"type": "string"}}},
    "required": ["captions", "hashtags_broad", "hashtags_niche", "hashtags_community", "alt_text", "first_comment",
                 "cover_texts"]}


def clip_text(p, c):
    if c.get("i0") is None:
        return c.get("caption", "")
    return " ".join(w["w"] for w in p.words[c["i0"]:c["i1"] + 1])


def instagram_package(p, s, c):
    cs.log(f"[instagram] writing the kit for “{c.get('title', '')}”...", 0.2)
    ctx = (f"{profile_text()}Reel hook: {c.get('title', '')}\nTopic: {c.get('topic', '')}\n"
           f"What is said in the Reel:\n{clip_text(p, c)}")
    d = llm_json(s, IG_PROMPT.format(language=language_name(p)), ctx, IG_SCHEMA)
    tags = lambda xs: [x.strip().lstrip("#").replace(" ", "") for x in xs if x.strip()]
    c["ig"] = {"captions": [x.strip() for x in d["captions"] if x.strip()][:5],
               "hashtags": {"broad": tags(d["hashtags_broad"]), "niche": tags(d["hashtags_niche"]),
                            "community": tags(d["hashtags_community"])},
               "alt_text": d["alt_text"].strip(), "first_comment": d["first_comment"].strip(),
               "cover_texts": [x.strip() for x in d["cover_texts"] if x.strip()][:4], "chat": [],
               **{k: v for k, v in c.get("ig", {}).items() if k in ("cover",)}}
    p.save()
    cs.log("[instagram] done", 1.0)
    return c["ig"]


def instagram_refine(p, s, c, message):
    ig = c.setdefault("ig", {})
    cs.log("[instagram] refining captions...", 0.3)
    d = refine(s, "Instagram Reel captions", language_name(p), ig.get("captions", []), ig.get("chat", []), message,
               f"Reel: {c.get('title', '')}\n{clip_text(p, c)}")
    ig["captions"] = [x.strip() for x in d["items"] if x.strip()][:5] or ig.get("captions", [])
    ig.setdefault("chat", []).extend([{"role": "you", "content": message}, {"role": "ai", "content": d["note"]}])
    p.save()
    return ig


# ---------------------------------------------------------------- frames: find good ones

def grab(video, t, width=None):
    vf = ["-vf", f"scale={width}:-2"] if width else []
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0, t):.3f}", "-i", str(video), "-frames:v", "1", *vf,
                          "-f", "image2", "-c:v", "png", "-"], capture_output=True).stdout
    from io import BytesIO
    return Image.open(BytesIO(raw)).convert("RGB")


def faces_in(img):
    """[(x, y, w, h, score)] in img pixels (YuNet on a 640-wide copy)."""
    import cv2
    w0 = 640
    small = img.resize((w0, int(img.height * w0 / img.width)))
    arr = cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR)
    det = cv2.FaceDetectorYN.create(str(cs.FACE_MODEL), "", (arr.shape[1], arr.shape[0]), 0.7, 0.3, 20)
    _, faces = det.detect(arr)
    k = img.width / w0
    return [(f[0] * k, f[1] * k, f[2] * k, f[3] * k, float(f[14])) for f in (faces if faces is not None else [])]


def frame_candidates(p, start=0.0, end=None, n=6, tag="cand"):
    """Pick n good frames in [start, end]: a big, confident, sharp face; spread out in time."""
    import cv2
    video = Path(p.data["video"])
    end = end or p.data["info"]["duration"]
    step = max(0.5, (end - start) / 60)
    w0 = 480
    proc = subprocess.run(["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                           "-i", str(video), "-vf", f"fps={1 / step:.5f},scale={w0}:-2", "-f", "rawvideo",
                           "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
    info = p.data["info"]
    h0 = int(round(w0 * info["h"] / info["w"] / 2)) * 2
    size = w0 * h0 * 3
    det = cv2.FaceDetectorYN.create(str(cs.FACE_MODEL), "", (w0, h0), 0.7, 0.3, 20)
    scored = []
    for i in range(len(proc) // size):
        fr = np.frombuffer(proc[i * size:(i + 1) * size], np.uint8).reshape(h0, w0, 3)
        _, faces = det.detect(fr)
        sharp = cv2.Laplacian(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        face = max(((f[3] / h0) * float(f[14]) for f in faces), default=0) if faces is not None else 0
        scored.append((face * 3 + min(sharp, 400) / 400 * 0.5, start + i * step))
    pool = []
    gap = max(2.0, (end - start) / (n * 2.5))
    for sc, t in sorted(scored, reverse=True):
        if all(abs(t - u) >= gap for u in pool):
            pool.append(t)
        if len(pool) >= n * 2:
            break
    picks = rank_expressions(p, video, pool)[:n] if p.settings.vision else pool[:n]
    out_dir = p.dir / "studio"
    out_dir.mkdir(exist_ok=True)
    res = []
    for k, t in enumerate(sorted(picks)):
        f = out_dir / f"{tag}_{k}.jpg"
        grab(video, t, 480).save(f, quality=85)
        res.append({"t": round(t, 2), "file": f"studio/{f.name}", "v": time.time()})
    return res


EXPR_SCHEMA = {"type": "object", "properties": {"eyes_open": {"type": "boolean"}, "expressive": {"type": "integer"},
                                                 "blurry": {"type": "boolean"}},
               "required": ["eyes_open", "expressive", "blurry"]}


def rank_expressions(p, video, times):
    """Order frames by how good the face looks (vision model): eyes open, expressive, not blurry."""
    import base64
    s = p.settings
    url = s.ollama.rstrip("/")
    try:
        cs.ensure_model_on_gpu(url, s.vision_model, 4096)
    except SystemExit as e:
        cs.log(f"[frames] skipping the expression check: {e}")
        return times
    rated = []
    for t in times:
        from io import BytesIO
        buf = BytesIO()
        grab(video, t, 512).save(buf, "JPEG", quality=85)
        try:
            r = cs._post(url + "/api/chat", {"model": s.vision_model, "stream": False, "format": EXPR_SCHEMA, "think": False,
                                             "options": {"temperature": 0, "num_ctx": 4096},
                                             "messages": [{"role": "user", "images": [base64.b64encode(buf.getvalue()).decode()],
                                                           "content": "Rate the main person's face for a video thumbnail. "
                                                                      "eyes_open: both eyes clearly open. expressive: 0-10, how "
                                                                      "engaging and expressive the face is. blurry: motion blur."}]}, 60)
            v = json.loads(r["message"]["content"])
            rated.append(((1 if v["eyes_open"] else -10) + int(v["expressive"]) - (5 if v["blurry"] else 0), t))
        except Exception:  # noqa: BLE001
            rated.append((0, t))
    return [t for _, t in sorted(rated, key=lambda x: -x[0])]


# ---------------------------------------------------------------- composing images

def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def fit_text(text, box_w, box_h, max_size, max_lines=3):
    """Largest font size (and its lines) that fits text in the box."""
    words = text.split()
    for size in range(max_size, 20, -4):
        font = ImageFont.truetype(FONT, size)
        lines, cur = [], ""
        for w in words:
            trial = f"{cur} {w}".strip()
            if font.getlength(trial) <= box_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        lh = int(size * 1.08)
        if len(lines) <= max_lines and all(font.getlength(l) <= box_w for l in lines) and lh * len(lines) <= box_h:
            return font, lines, lh
    font = ImageFont.truetype(FONT, 24)
    return font, [text], 26


def draw_title(img, text, box, highlight="#FFE500", style="stroke", align="left"):
    """Big bold text in box=(x, y, w, h). The longest word gets the highlight color."""
    x, y, w, h = box
    text = text.upper()
    font, lines, lh = fit_text(text, w, h, max_size=int(h / 1.6))
    d = ImageDraw.Draw(img)
    hl = max(text.split(), key=len)
    total = lh * len(lines)
    yy = y + (h - total) // 2
    size = font.size
    for line in lines:
        lw = font.getlength(line)
        xx = x if align == "left" else x + (w - lw) / 2
        if style == "box":
            pad = size * 0.18
            d.rounded_rectangle([xx - pad, yy - pad * 0.4, xx + lw + pad, yy + lh - pad * 0.2], radius=size * 0.18,
                                fill=hex_rgb(highlight))
        cx = xx
        for word in line.split(" "):
            color = (0, 0, 0) if style == "box" else (hex_rgb(highlight) if word == hl else (255, 255, 255))
            d.text((cx, yy), word, font=font, fill=color,
                   stroke_width=0 if style == "box" else max(3, int(size * 0.09)), stroke_fill=(0, 0, 0))
            cx += font.getlength(word + " ")
        yy += lh
    return img


def cover_crop(img, aspect, face=None):
    """Crop img to aspect (w/h), keeping the face in frame (upper third)."""
    W, H = img.size
    if W / H > aspect:
        cw, ch = int(H * aspect), H
    else:
        cw, ch = W, int(W / aspect)
    fx = face[0] + face[2] / 2 if face else W / 2
    fy = face[1] + face[3] / 2 if face else H * 0.4
    x = int(min(max(fx - cw / 2, 0), W - cw))
    y = int(min(max(fy - ch * 0.38, 0), H - ch))
    return img.crop((x, y, x + cw, y + ch))


def punch(img, sat=1.25, con=1.12):
    return ImageEnhance.Contrast(ImageEnhance.Color(img).enhance(sat)).enhance(con)


def make_thumbnail(p, t, text, style="right", highlight=None):
    """YouTube thumbnail 1280x720 from the frame at t. style: right (person right, text left), left, or box."""
    s = p.settings
    hl = highlight or s.highlight
    frame = grab(Path(p.data["video"]), t)
    faces = sorted(faces_in(frame), key=lambda f: -f[3])
    face = faces[0] if faces else None
    W, H = 1280, 720
    bg = punch(frame.copy())
    bg = cover_crop(bg, W / H, face).resize((W, H), Image.LANCZOS)
    portrait = frame.width / frame.height < 1.2
    if portrait or style in ("right", "left"):
        canvas = bg.filter(ImageFilter.GaussianBlur(24))
        canvas = ImageEnhance.Brightness(canvas).enhance(0.45)
        person = punch(cover_crop(frame, 0.75, face)).resize((540, 720), Image.LANCZOS)
        px = W - 540 - 20 if style != "left" else 20
        canvas.paste(person, (px, 0))
        tx = 50 if style != "left" else 600
        draw_title(canvas, text, (tx, 60, 620, 600), hl, "box" if style == "box" else "stroke")
    else:
        canvas = bg
        shade = Image.new("L", (W, H), 0)
        ImageDraw.Draw(shade).rectangle([0, H * 0.55, W, H], fill=170)
        canvas = Image.composite(Image.new("RGB", (W, H), (0, 0, 0)), canvas, shade.filter(ImageFilter.GaussianBlur(60)))
        draw_title(canvas, text, (60, int(H * 0.58), W - 120, int(H * 0.36)), hl, "box", align="center")
    out_dir = p.dir / "studio"
    out_dir.mkdir(exist_ok=True)
    n = len(list(out_dir.glob("thumb_*.jpg"))) + 1
    f = out_dir / f"thumb_{n:02d}.jpg"
    canvas.save(f, quality=92)
    yt = p.data.setdefault("youtube", {})
    yt.setdefault("thumbs", []).append({"file": f"studio/{f.name}", "text": text, "style": style, "t": t})
    p.save()
    return yt["thumbs"][-1]


def make_cover(p, c, t, text, style="stroke"):
    """Instagram Reels cover 1080x1920 from the source frame at t (absolute seconds); the title sits in
    the centre 3:4 area so it survives the profile-grid crop."""
    s = Settings_for(p, c)
    frame = grab(Path(p.data["video"]), t)
    faces = sorted(faces_in(frame), key=lambda f: -f[3])
    img = punch(cover_crop(frame, 1080 / 1920, faces[0] if faces else None)).resize((1080, 1920), Image.LANCZOS)
    if text:
        grid_top = (1920 - 1440) // 2                       # profile grid shows the middle 1080x1440
        shade = Image.new("L", img.size, 0)
        ImageDraw.Draw(shade).rectangle([0, grid_top + 760, 1080, grid_top + 1440], fill=150)
        img = Image.composite(Image.new("RGB", img.size, (0, 0, 0)), img, shade.filter(ImageFilter.GaussianBlur(80)))
        draw_title(img, text, (70, grid_top + 860, 940, 520), s.highlight, style, align="center")
    name = Path(c.get("file") or f"clip{c['id']}.mp4").stem + "_cover.jpg"
    img.save(p.dir / name, quality=92)
    c.setdefault("ig", {})["cover"] = {"file": name, "t": t, "text": text, "v": time.time()}
    p.save()
    return c["ig"]["cover"]


def Settings_for(p, c):
    return cs.Settings.from_dict({**p.data.get("settings", {}), **c.get("overrides", {})})
