#!/usr/bin/env -S uv run --project /home/jt/claude/clipsmith python
"""clipsmith: our own Opus Clip. Turns a long video into ready-to-post vertical shorts.

video -> Whisper (word timestamps) -> local LLM picks the best moments -> face-tracked 9:16 crop that
follows whoever is talking -> dead air trimmed, animated word-by-word captions, hook title, loudness
normalized -> 1080x1920 MP4s plus a gallery page with titles, captions and hashtags.

Everything runs on this machine: faster-whisper on the GPU, Ollama for the editor brain, ffmpeg/NVENC.
"""
import argparse
import gc
import hashlib
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
FACE_MODEL = ASSETS / "face_detection_yunet_2023mar.onnx"
FONT_NAME = "Montserrat Thin ExtraBold"   # full name inside assets/Montserrat-ExtraBold.ttf
OUT_W, OUT_H = 1080, 1920


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def fmt_ts(t):
    m, s = divmod(int(t), 60)
    return f"{m}:{s:02d}"


def parse_ts(s):
    """'83', '1:23', '1:02:03' or '83.5' -> seconds."""
    parts = [float(p) for p in s.strip().split(":")]
    t = 0.0
    for p in parts:
        t = t * 60 + p
    return t


# ---------------------------------------------------------------- probing

def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
                          "-show_format", str(path)], capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        sys.exit(f"{path}: no video stream")
    w, h = int(v["width"]), int(v["height"])
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = int(v.get("tags", {}).get("rotate", rot))
    if abs(rot) % 180 == 90:
        w, h = h, w
    num, den = (int(x) for x in v.get("avg_frame_rate", "30/1").split("/"))
    fps = num / den if den else 30.0
    if not 1 <= fps <= 240:
        fps = 30.0
    return {"w": w, "h": h, "fps": fps, "duration": float(info["format"]["duration"]),
            "audio": any(s["codec_type"] == "audio" for s in info["streams"])}


# ---------------------------------------------------------------- transcription

def load_cuda12_libs():
    """ctranslate2 (faster-whisper) is built for CUDA 12. Load cuBLAS/cuDNN from this venv's nvidia
    wheels by full path so nothing needs LD_LIBRARY_PATH."""
    import ctypes
    import glob
    import site
    pending = []
    for sp in site.getsitepackages():
        pending += sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib", "*.so*")))
    for _ in range(4):  # some libraries need others loaded first; retry the ones that failed
        failed = []
        for p in pending:
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(p)
        if len(failed) in (0, len(pending)):
            break
        pending = failed


def transcribe(video, work, model_name, language, force):
    st = video.stat()
    key = hashlib.sha1(f"{video.resolve()}|{st.st_size}|{st.st_mtime}|{model_name}|{language}".encode()).hexdigest()[:12]
    cache = work / f"transcript-{key}.json"
    if cache.exists() and not force:
        log(f"[1/4] transcript: cached ({cache.name})")
        return json.loads(cache.read_text())

    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(pcm, np.float32)
    load_cuda12_libs()
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    t0 = time.time()
    # GPU only, never the CPU. The GPU is shared (Ollama, ComfyUI...), so retry with smaller batches.
    attempts = [("cuda", "float16", 16), ("cuda", "float16", 4)]
    for device, ctype, batch in attempts:
        try:
            model = WhisperModel(model_name, device=device, compute_type=ctype)
            pipe = BatchedInferencePipeline(model=model)
            log(f"[1/4] transcribing with Whisper {model_name} ({device}, {ctype})...")
            segments, tinfo = pipe.transcribe(audio, batch_size=batch, word_timestamps=True, vad_filter=True,
                                              language=language)
            words = []
            for seg in segments:
                for w in seg.words or []:
                    text = w.word.strip()
                    if text:
                        words.append({"w": text, "s": round(w.start, 3), "e": round(w.end, 3)})
            break
        except Exception as e:  # noqa: BLE001
            if batch == attempts[-1][2]:
                sys.exit(f"Whisper can't run on the GPU ({str(e)[:100]}). Free some VRAM (check "
                         "`nvidia-smi` and `ollama ps`) and try again. clipsmith never falls back to the CPU.")
            log(f"[1/4] GPU attempt failed ({str(e)[:80]}); retrying with a smaller batch")
        finally:
            pipe = model = None
            gc.collect()
    data = {"language": tinfo.language, "words": words}
    cache.write_text(json.dumps(data))
    log(f"[1/4] transcribed {len(words)} words in {time.time() - t0:.0f}s (language: {tinfo.language})")
    return data


def sentences_from_words(words):
    """Group words into sentence-ish lines: split on . ? ! or a long pause, and keep lines short."""
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(i)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["s"] - w["e"]) if nxt else 99
        end_punct = w["w"][-1] in ".?!"
        if end_punct or gap > 0.9 or (len(cur) >= 30 and (w["w"][-1] in ",;:" or gap > 0.3)) or len(cur) >= 45:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [{"i0": c[0], "i1": c[-1], "s": words[c[0]]["s"], "e": words[c[-1]]["e"],
             "text": " ".join(words[j]["w"] for j in c)} for c in out]


# ---------------------------------------------------------------- picking moments (the LLM)

EDITOR_PROMPT = """You are a top short-form video editor who finds viral moments in long videos for \
Instagram Reels. You get a timestamped transcript, one numbered line per sentence. Pick the best \
self-contained moments to cut into vertical shorts.

What makes a great short:
- HOOK: the first line grabs attention within 3 seconds (a bold claim, a question, a surprising fact, \
conflict, a strong emotion, a punchline setup). It must make sense with no earlier context: never start \
on "And", "So", "But", "That's why", or on a pronoun that refers to something said before.
- PAYOFF: it ends on a complete thought: the answer, punchline, lesson or twist. Never stop mid-story.
- One clear idea per clip. Story, insight, hot take, funny exchange, emotional peak, or useful tip.
- Prefer moments people would rewatch, share or comment on.

Rules:
- Each clip spans whole lines: start_line to end_line (inclusive), using the line numbers shown.
- Each clip lasts between {min_s} and {max_s} seconds (use the timestamps). Clips must not overlap.
- Return up to {n} clips, best first. Fewer is fine if the material is weak; never pad with filler.
- score: honest 0-100 estimate of viral potential (90+ only for exceptional moments).
- title: on-screen hook text for the first seconds, max 8 words, punchy, no hashtags, no emojis.
- caption: Instagram caption, 1-2 short sentences that tease the clip, plus a light call to action.
- hashtags: 5-8 relevant hashtags, without the # sign.
- why: one sentence on why this moment works.
- Write title, caption and hashtags in the same language as the transcript ({language}), in the \
speaker's own voice and vocabulary. Write "why" in English."""

CLIP_SCHEMA = {
    "type": "object",
    "properties": {"clips": {"type": "array", "items": {
        "type": "object",
        "properties": {"start_line": {"type": "integer"}, "end_line": {"type": "integer"},
                       "score": {"type": "integer"}, "title": {"type": "string"},
                       "caption": {"type": "string"}, "hashtags": {"type": "array", "items": {"type": "string"}},
                       "why": {"type": "string"}},
        "required": ["start_line", "end_line", "score", "title", "caption", "hashtags", "why"]}}},
    "required": ["clips"],
}


def ollama_chat(url, model, system, user, think):
    body = {"model": model, "stream": False, "format": CLIP_SCHEMA, "think": think, "keep_alive": "5m",
            "options": {"temperature": 0.4, "num_ctx": 32768},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    for attempt in range(2):
        req = urllib.request.Request(url.rstrip("/") + "/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.loads(r.read())["message"]["content"]
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")
            if attempt == 0 and "think" in msg:   # model doesn't support the think switch
                body.pop("think")
                continue
            raise SystemExit(f"Ollama error {e.code}: {msg}")
        except urllib.error.URLError as e:
            raise SystemExit(f"Can't reach Ollama at {url} ({e.reason}). Is `ollama serve` running?")


def ollama_unload(url, model):
    try:
        req = urllib.request.Request(url.rstrip("/") + "/api/generate",
                                     json.dumps({"model": model, "keep_alive": 0}).encode(),
                                     {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
    except Exception:  # noqa: BLE001
        pass


def fit_length(c, sents, min_s, max_s):
    """Trim or extend a clip (in whole lines) to fit the allowed length. Returns None if hopeless."""
    a, b = c["a"], c["b"]
    while b > a and sents[b]["e"] - sents[a]["s"] > max_s:
        b -= 1
    while b + 1 < len(sents) and sents[b]["e"] - sents[a]["s"] < min_s \
            and sents[b + 1]["e"] - sents[a]["s"] <= max_s:
        b += 1
    dur = sents[b]["e"] - sents[a]["s"]
    if dur < min_s * 0.75 or dur > max_s * 1.1:
        return None
    return {**c, "a": a, "b": b}


def pick_clips(sents, args):
    """Ask the LLM for candidate moments, window by window for long videos; then rank and de-overlap."""
    window_s, overlap_s = 15 * 60, 90
    windows, start = [], 0
    while start < len(sents):
        end = start
        while end + 1 < len(sents) and sents[end + 1]["e"] - sents[start]["s"] < window_s:
            end += 1
        windows.append((start, end))
        if end + 1 >= len(sents):
            break
        nxt = end
        while nxt > start and sents[end]["e"] - sents[nxt]["s"] < overlap_s:
            nxt -= 1
        start = max(nxt, start + 1)

    per_window = max(3, math.ceil(args.count * 1.5 / len(windows)) + 1)
    system = EDITOR_PROMPT.format(min_s=args.min, max_s=args.max, n=per_window, language=args.spoken)
    cands = []
    for wi, (a, b) in enumerate(windows):
        lines = "\n".join(f"[{i}] ({fmt_ts(sents[i]['s'])}) {sents[i]['text']}" for i in range(a, b + 1))
        t0 = time.time()
        log(f"[2/4] {args.model} reading transcript part {wi + 1}/{len(windows)} "
            f"({fmt_ts(sents[a]['s'])}-{fmt_ts(sents[b]['e'])})...")
        raw = ollama_chat(args.ollama, args.model, system, f"Transcript:\n{lines}", args.think)
        try:
            got = json.loads(raw).get("clips", [])
        except json.JSONDecodeError:
            log(f"[2/4] couldn't parse the model's answer, skipping this part: {raw[:200]}")
            got = []
        log(f"[2/4]   {len(got)} candidates in {time.time() - t0:.0f}s")
        for c in got:
            lo, hi = sorted((int(c["start_line"]), int(c["end_line"])))
            if not (a <= lo <= b and a <= hi <= b):
                continue
            c = fit_length({"a": lo, "b": hi, "score": max(0, min(100, int(c["score"]))),
                            "title": c["title"].strip().strip('"'), "caption": c["caption"].strip(),
                            "hashtags": [h.strip().lstrip("#").replace(" ", "") for h in c["hashtags"] if h.strip()],
                            "why": c["why"].strip()}, sents, args.min, args.max)
            if c:
                cands.append(c)
    ollama_unload(args.ollama, args.model)   # hand the GPU back

    chosen = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        s, e = sents[c["a"]]["s"], sents[c["b"]]["e"]
        if all(min(e, sents[o["b"]]["e"]) - max(s, sents[o["a"]]["s"]) < 0.25 * min(e - s, 1e9)
               for o in chosen):
            chosen.append(c)
        if len(chosen) >= args.count:
            break
    return chosen


# ---------------------------------------------------------------- clip timing and dead-air trimming

def clip_bounds(words, i0, i1, duration):
    s = words[i0]["s"] - 0.15
    if i0 > 0:
        s = max(s, words[i0 - 1]["e"] + 0.02)
    e = words[i1]["e"] + 0.4
    if i1 + 1 < len(words):
        e = min(e, max(words[i1]["e"] + 0.1, words[i1 + 1]["s"] - 0.05))
    return max(0.0, s), min(duration, e)


def keep_intervals(words, i0, i1, cs, ce, max_gap):
    """Pieces of [cs, ce] to keep (clip-relative seconds), cutting pauses longer than max_gap."""
    if max_gap <= 0:
        return [(0.0, ce - cs)]
    pieces, a = [], 0.0
    for i in range(i0, i1):
        gap = words[i + 1]["s"] - words[i]["e"]
        if gap > max_gap:
            pieces.append((a, words[i]["e"] - cs + 0.12))
            a = words[i + 1]["s"] - cs - 0.1
    pieces.append((a, ce - cs))
    return [(max(0.0, p), min(ce - cs, q)) for p, q in pieces if q - p > 0.05]


class Timeline:
    """Maps clip-relative source time to output time after the cuts."""

    def __init__(self, pieces):
        self.pieces = pieces
        self.offsets, acc = [], 0.0
        for a, b in pieces:
            self.offsets.append(acc)
            acc += b - a
        self.length = acc

    def __call__(self, t):
        for (a, b), off in zip(self.pieces, self.offsets):
            if t < a:
                return off
            if t <= b:
                return off + t - a
        return self.length


# ---------------------------------------------------------------- reframing (who to look at)

def analyze_faces(video, info, cs, dur, sample_fps=6.0):
    """Sample the clip, find faces, and estimate who is talking from mouth motion.
    Returns (samples, analysis_width) where samples = [(t, [(cx, cy, w, h, mouth_motion)])]."""
    import cv2
    aw = 640
    ah = int(round(aw * info["h"] / info["w"] / 2)) * 2
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{cs:.3f}", "-t", f"{dur:.3f}", "-i", str(video),
                             "-vf", f"fps={sample_fps},scale={aw}:{ah}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                            stdout=subprocess.PIPE)
    det = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (aw, ah), 0.7, 0.3, 50)
    samples, prev, n = [], None, 0
    fsize = aw * ah * 3
    while True:
        buf = proc.stdout.read(fsize)
        if len(buf) < fsize:
            break
        frame = np.frombuffer(buf, np.uint8).reshape(ah, aw, 3)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
        _, faces = det.detect(frame)
        found = []
        for f in (faces if faces is not None else []):
            x, y, w, h = (float(v) for v in f[:4])
            if h < ah * 0.06:          # tiny background faces are noise
                continue
            mx0, mx1 = sorted((f[10], f[12]))
            my = (f[11] + f[13]) / 2
            box = lambda x0, y0, x1, y1: (slice(max(0, int(y0)), min(ah, int(y1))),
                                          slice(max(0, int(x0)), min(aw, int(x1))))
            mouth = box(mx0 - 0.12 * w, my - 0.12 * h, mx1 + 0.12 * w, my + 0.22 * h)
            face = box(x, y, x + w, y + h)
            motion = 0.0
            if prev is not None and gray[mouth].size and gray[face].size:
                m = np.abs(gray[mouth] - prev[mouth]).mean()
                head = np.abs(gray[face] - prev[face]).mean()
                motion = max(0.0, m - 0.7 * head)
            found.append((x + w / 2, y + h / 2, w, h, motion))
        samples.append((n / sample_fps, found))
        prev, n = gray, n + 1
    proc.wait()
    return samples, aw, ah


def hook_margin(samples, ah, y_scale, y_off, lines):
    """Where to put the hook title (ASS top margin) so it doesn't cover the face: above the head if
    there's room, otherwise just under the chin, otherwise the default spot."""
    faces = [max(fs, key=lambda f: f[3]) for _, fs in samples if fs]
    default = 250
    if len(faces) < 0.3 * max(1, len(samples)):
        return default
    top = np.median([cy - 0.75 * h for cx, cy, w, h, _ in faces]) * y_scale + y_off   # incl. hair
    bottom = np.median([cy + 0.6 * h for cx, cy, w, h, _ in faces]) * y_scale + y_off
    height = lines * 105 + 40
    if top - 40 >= default + height:
        return default
    if bottom + 30 + height <= 1230:   # stay clear of the captions (top edge ~1250)
        return int(bottom + 30)
    return default


def plan_camera(samples, aw, crop_w_a, sample_fps=6.0):
    """Decide where the 9:16 window sits over time. Follows the active speaker, hard-cuts when the
    speaker changes (like a real editor), and glides smoothly otherwise.
    Returns (times, centers_in_analysis_px, cut_indices) or None if there are too few faces."""
    if not samples or sum(1 for _, f in samples if f) < 0.35 * len(samples):
        return None
    tracks, nid = [], 0       # each: {id, cx, h, act, last}
    per_sample = []           # [(track_id, cx, score)]
    for t, faces in samples:
        here = []
        for cx, cy, w, h, motion in faces:
            best = None
            for tr in tracks:
                d = abs(tr["cx"] - cx)
                if t - tr["last"] < 1.5 and d < 0.6 * max(w, tr["w"]) and (best is None or d < best[0]):
                    best = (d, tr)
            if best is None:
                tr = {"id": nid, "cx": cx, "w": w, "h": h, "act": 0.0, "last": t}
                nid += 1
                tracks.append(tr)
            else:
                tr = best[1]
            tr.update(cx=cx, w=w, h=h, last=t, act=0.75 * tr["act"] + 0.25 * motion)
            here.append(tr)
        top = max((tr["act"] for tr in here), default=0) or 1.0
        per_sample.append([(tr["id"], tr["cx"], (tr["h"] / aw) * (0.3 + tr["act"] / top)) for tr in here])

    # pick a target per sample, switching only when someone else clearly wins for a while
    cur, challenger, streak, missing = None, None, 0, 0
    hold, patience = int(1.0 * sample_fps), int(0.7 * sample_fps)
    xs, ids = [], []
    for faces in per_sample:
        scores = {tid: (cx, sc) for tid, cx, sc in faces}
        best = max(faces, key=lambda f: f[2]) if faces else None
        if cur not in scores:
            missing += 1
            if best and (cur is None or missing >= patience):
                cur, missing, challenger, streak = best[0], 0, None, 0
        else:
            missing = 0
            if best[0] != cur and best[2] > 1.3 * scores[cur][1]:
                streak = streak + 1 if challenger == best[0] else 1
                challenger = best[0]
                if streak >= hold:
                    cur, challenger, streak = best[0], None, 0
            else:
                challenger, streak = None, 0
        xs.append(scores[cur][0] if cur in scores else None)
        ids.append(cur)

    # fill gaps (face briefly lost) with the last known position
    first = next((x for x in xs if x is not None), aw / 2)
    for i in range(len(xs)):
        if xs[i] is None:
            xs[i] = xs[i - 1] if i else first
    first_id = next((t for t in ids if t is not None), None)
    ids = [first_id if t is None else t for t in ids]

    cuts = [i for i in range(1, len(ids)) if ids[i] != ids[i - 1]]
    bounds = [0] + cuts + [len(xs)]
    out = np.array(xs, float)
    dz = 0.08 * crop_w_a          # deadzone: small head moves don't move the camera
    win = max(1, int(0.8 * sample_fps))
    for a, b in zip(bounds, bounds[1:]):
        seg = out[a:b].copy()
        pos = seg[0]
        for k in range(len(seg)):
            d = seg[k] - pos
            if abs(d) > dz:
                pos += d - math.copysign(dz, d)
            seg[k] = pos
        if len(seg) > 2:          # gentle zero-phase smoothing
            pad = np.pad(seg, win, mode="edge")
            seg = np.convolve(pad, np.ones(2 * win + 1) / (2 * win + 1), mode="same")[win:-win]
        out[a:b] = seg
    times = [t for t, _ in samples]
    return times, out, cuts


def write_crop_commands(path, times, centers, cuts, scale, crop_w, src_w, fps, dur):
    """ffmpeg sendcmd script moving crop@rf frame by frame (hard jumps at speaker cuts)."""
    times = np.array(times)
    cut_times = [times[i] for i in cuts]
    seg_of = lambda t: sum(1 for ct in cut_times if t >= ct)
    lines, last = [], None
    for k in range(int(dur * fps) + 2):
        t = k / fps
        seg = seg_of(t)
        lo = 0 if seg == 0 else cuts[seg - 1]
        hi = cuts[seg] if seg < len(cuts) else len(times)
        c = np.interp(t, times[lo:hi], centers[lo:hi])
        x = int(round(min(max(c * scale - crop_w / 2, 0), src_w - crop_w) / 2) * 2)
        if x != last:
            lines.append(f"{t:.4f} crop@rf x {x};")
            last = x
    Path(path).write_text("\n".join(lines) + "\n")
    first = int(lines[0].split(" x ")[1].rstrip(";")) if lines else 0
    return first


# ---------------------------------------------------------------- captions (ASS)

def ass_time(t):
    t = max(0.0, t)
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def ass_text(s):
    return s.replace("\\", "").replace("{", "(").replace("}", ")")


def wrap(text, width):
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


STYLES = {
    #        font size, highlight (ASS &HBBGGRR), max words per caption, uppercase
    "bold": dict(size=100, hi="&H0000E5FF", words=3, upper=True),
    "green": dict(size=100, hi="&H0014FF39", words=3, upper=True),
    "clean": dict(size=70, hi="&H00FFFFFF", words=5, upper=False),
}


def write_ass(path, words, tl, cs, title, style, hook_s, hook_v=250):
    st = STYLES[style] if style != "none" else None
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {OUT_W}
PlayResY: {OUT_H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{FONT_NAME},{st['size'] if st else 80},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,7,3,2,60,60,560,1
Style: Hook,{FONT_NAME},84,&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,18,0,8,90,90,250,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    if title and hook_s > 0:
        txt = "\\N".join(ass_text(l) for l in wrap(title, 20))
        ev.append(f"Dialogue: 1,{ass_time(0)},{ass_time(hook_s)},Hook,,0,0,0,,"
                  f"{{\\an8\\pos({OUT_W // 2},{hook_v})\\fad(150,250)\\fscx90\\fscy90\\t(0,180,\\fscx100\\fscy100)}}{txt}")
    if st:
        ws = [{"w": w["w"], "s": tl(w["s"] - cs), "e": tl(w["e"] - cs)} for w in words]
        chunks, cur = [], []
        for i, w in enumerate(ws):
            cur.append(w)
            nxt = ws[i + 1] if i + 1 < len(ws) else None
            chars = sum(len(x["w"]) + 1 for x in cur)
            if (not nxt or len(cur) >= st["words"] or chars + len(nxt["w"]) > 18 + 6 * (st["words"] > 3)
                    or w["w"][-1] in ".?!,;:" or nxt["s"] - w["e"] > 0.35):
                chunks.append(cur)
                cur = []
        for ci, ch in enumerate(chunks):
            end = min(ch[-1]["e"] + 0.5, chunks[ci + 1][0]["s"]) if ci + 1 < len(chunks) else ch[-1]["e"] + 0.5
            for k, w in enumerate(ch):
                a = w["s"] if k else min(w["s"], ch[0]["s"])
                b = ch[k + 1]["s"] if k + 1 < len(ch) else end
                if b - a < 0.02:
                    continue
                parts = []
                for j, x in enumerate(ch):
                    t = ass_text(x["w"].upper() if st["upper"] else x["w"])
                    parts.append(f"{{\\c{st['hi']}\\fscx108\\fscy108}}{t}{{\\c&H00FFFFFF&\\fscx100\\fscy100}}"
                                 if j == k and style != "clean" else t)
                pop = "{\\fscx85\\fscy85\\t(0,90,\\fscx100\\fscy100)}" if k == 0 else ""
                ev.append(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Cap,,0,0,0,,{pop}{' '.join(parts)}")
    Path(path).write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- rendering

def fesc(p):
    """Escape a path for use inside an ffmpeg filtergraph option."""
    return str(p).replace("\\", "\\\\\\\\").replace(":", "\\\\:").replace("'", "\\\\\\'").replace(",", "\\,")


def nvenc_ok():
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
                        "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True)
    return r.returncode == 0


def render(video, info, clip, words, out_path, work, args, encoder):
    cs, ce = clip["start"], clip["end"]
    dur = ce - cs
    spoken = clip["i0"] is not None
    pieces = keep_intervals(words, clip["i0"], clip["i1"], cs, ce, args.max_gap if spoken else 0)
    tl = Timeline(pieces)
    fps = min(info["fps"], 60.0)
    tag = out_path.stem
    W, H = info["w"], info["h"]

    # where to look
    layout, crop_w = args.layout, int(round(H * 9 / 16 / 2)) * 2
    plan, samples = None, None
    if W / H <= 9 / 16 + 0.01:
        layout = "full"                       # already vertical
    elif layout in ("auto", "track"):
        samples, aw, ah = analyze_faces(video, info, cs, dur)
        plan = plan_camera(samples, aw, crop_w * aw / W)
        if plan is None:
            if layout == "track":
                plan = ([0.0], np.array([aw / 2]), [])
            else:
                layout = "fit"
        if plan is not None:
            layout = "track"

    if layout == "track":
        cmd_file = work / f"{tag}.cmd"
        x0 = write_crop_commands(cmd_file, plan[0], plan[1], plan[2], W / 640, crop_w, W, fps, dur)
        vchain = (f"[0:v]sendcmd=f={fesc(cmd_file.name)},crop@rf=w={crop_w}:h={H}:x={x0}:y=0,"
                  f"scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1,fps={fps:.4f}[v0]")
        how = f"speaker tracking, {len(plan[2])} camera cuts"
    elif layout == "fit":
        vchain = (f"[0:v]split[bgi][fgi];[bgi]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
                  f"crop={OUT_W}:{OUT_H},gblur=sigma=40,eq=brightness=-0.12[bg];"
                  f"[fgi]scale={OUT_W}:-2:flags=lanczos[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={fps:.4f}[v0]")
        how = "fit with blurred background (few faces)"
    elif layout == "center":
        vchain = (f"[0:v]crop={crop_w}:{H}:(iw-{crop_w})/2:0,scale={OUT_W}:{OUT_H}:flags=lanczos,"
                  f"setsar=1,fps={fps:.4f}[v0]")
        how = "center crop"
    else:
        vchain = (f"[0:v]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=lanczos,"
                  f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps:.4f}[v0]")
        how = "already vertical"

    hook_lines = len(wrap(clip["title"], 20))
    hook_v = 250
    if layout == "full" and not args.no_hook:
        samples, aw, ah = analyze_faces(video, info, cs, min(dur, 8.0))
        sc = min(OUT_W / aw, OUT_H / ah)
        hook_v = hook_margin(samples, ah, sc, (OUT_H - ah * sc) / 2, hook_lines)
    elif layout == "track" and samples:
        hook_v = hook_margin(samples, None, OUT_H / (640 * H / W), 0, hook_lines)
    ass = work / f"{tag}.ass"
    write_ass(ass, words[clip["i0"]:clip["i1"] + 1] if spoken else [], tl, cs, clip["title"] if not args.no_hook else "",
              args.captions, args.hook_seconds, hook_v)

    n = len(pieces)
    g = [vchain]
    g.append(f"[v0]split={n}" + "".join(f"[vs{i}]" for i in range(n)) if n > 1 else "[v0]null[vs0]")
    has_a = info["audio"]
    if has_a:
        g.append(f"[0:a]asplit={n}" + "".join(f"[as{i}]" for i in range(n)) if n > 1 else "[0:a]anull[as0]")
    cat = ""
    for i, (a, b) in enumerate(pieces):
        g.append(f"[vs{i}]trim=start={a:.4f}:end={b:.4f},setpts=PTS-STARTPTS[vt{i}]")
        cat += f"[vt{i}]"
        if has_a:
            fo = max(0.0, b - a - 0.012)
            g.append(f"[as{i}]atrim=start={a:.4f}:end={b:.4f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d=0.012,afade=t=out:st={fo:.4f}:d=0.012[at{i}]")
            cat += f"[at{i}]"
    g.append(f"{cat}concat=n={n}:v=1:a={1 if has_a else 0}[vc]" + ("[ac]" if has_a else ""))
    g.append(f"[vc]subtitles=f={fesc(ass.name)}:fontsdir={fesc(ASSETS)}[vout]")
    if has_a:
        g.append("[ac]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]")
    graph = work / f"{tag}.filter"
    graph.write_text(";\n".join(g))

    venc = (["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", "18", "-b:v", "0",
             "-maxrate", "25M", "-bufsize", "50M", "-spatial-aq", "1", "-temporal-aq", "1"]
            if encoder == "nvenc" else ["-c:v", "libx264", "-preset", "slow", "-crf", "17"])
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{cs:.3f}", "-t", f"{dur:.3f}", "-i", str(video.resolve()),
           "-filter_complex_script", graph.name, "-map", "[vout]"]
    if has_a:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    cmd += venc + ["-profile:v", "high", "-pix_fmt", "yuv420p", "-g", str(int(round(fps * 2))),
                   "-movflags", "+faststart", str(out_path.resolve())]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"ffmpeg failed for {tag}:\n{r.stderr[-3000:]}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{min(1.2, tl.length / 2):.2f}", "-i", str(out_path),
                    "-frames:v", "1", "-q:v", "3", str(out_path.with_suffix(".jpg"))], check=False)
    return {"layout": how, "length": round(tl.length, 2), "cut_s": round(dur - tl.length, 2),
            "render_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- gallery

def write_gallery(out_dir, video_name, clips):
    cards = []
    for c in clips:
        tags = " ".join("#" + h for h in c["hashtags"])
        post = f"{c['caption']}\n\n{tags}"
        cards.append(f"""<article>
  <video src="{html.escape(c['file'])}" poster="{html.escape(Path(c['file']).with_suffix('.jpg').name)}" controls preload="none" playsinline></video>
  <div class="meta">
    <div class="row"><span class="score s{c['score'] // 10 * 10}">{c['score']}</span><h2>{html.escape(c['title'])}</h2></div>
    <p class="sub">{fmt_ts(c['start'])}–{fmt_ts(c['end'])} in source · {c['length']:.0f}s · {html.escape(c['layout'])}</p>
    <p class="why">{html.escape(c['why'])}</p>
    <pre>{html.escape(post)}</pre>
    <div class="row"><button data-copy="{html.escape(post)}">Copy caption</button><a href="{html.escape(c['file'])}" download>Download</a></div>
  </div>
</article>""")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Clips · {html.escape(video_name)}</title>
<style>
:root{{--bg:#0e0f12;--card:#17191e;--fg:#eceef2;--mut:#9aa0ab;--acc:#ffe500}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}}
header{{padding:24px 16px 8px;max-width:1200px;margin:auto}} h1{{margin:0;font-size:22px}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:18px;padding:16px;max-width:1200px;margin:auto}}
article{{background:var(--card);border-radius:14px;overflow:hidden}}
video{{width:100%;aspect-ratio:9/16;background:#000;display:block}}
.meta{{padding:12px 14px 16px}} .row{{display:flex;gap:10px;align-items:center}} h2{{font-size:16px;margin:0}}
.score{{font-weight:800;border-radius:8px;padding:2px 8px;background:#333}} .s90,.s100{{background:var(--acc);color:#000}} .s80{{background:#9be15d;color:#000}} .s70{{background:#5dc3e1;color:#000}}
.sub,.why{{color:var(--mut);margin:6px 0}} pre{{white-space:pre-wrap;background:#0b0c0f;padding:10px;border-radius:8px;font:13px/1.4 system-ui}}
button,a{{background:#2a2d35;color:var(--fg);border:0;border-radius:8px;padding:7px 12px;font:inherit;cursor:pointer;text-decoration:none}}
</style></head><body><header><h1>{len(clips)} shorts from {html.escape(video_name)}</h1></header><main>
{chr(10).join(cards)}
</main><script>
document.querySelectorAll('button[data-copy]').forEach(b=>b.onclick=async()=>{{
 try{{await navigator.clipboard.writeText(b.dataset.copy);b.textContent='Copied!'}}catch(e){{b.textContent='Copy failed'}}
 setTimeout(()=>b.textContent='Copy caption',1500)}});
</script></body></html>"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


# ---------------------------------------------------------------- main

LANGS = {"en": "English", "pt": "Portuguese", "es": "Spanish", "fr": "French", "de": "German",
         "it": "Italian", "nl": "Dutch", "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}


def slug(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "clip"


def main():
    p = argparse.ArgumentParser(description="Turn a long video into vertical shorts for Instagram Reels.")
    p.add_argument("video", type=Path)
    p.add_argument("-o", "--out", type=Path, help="output folder (default: ./clips/<video name>)")
    p.add_argument("-n", "--count", type=int, default=5, help="how many shorts to make (default 5)")
    p.add_argument("--min", type=float, default=20, help="shortest clip in seconds (default 20)")
    p.add_argument("--max", type=float, default=60, help="longest clip in seconds (default 60)")
    p.add_argument("--clips", help="skip the AI and cut these ranges, e.g. '1:23-2:05,10:00-10:45'")
    p.add_argument("--model", default="qwen3.6:27b", help="Ollama model that picks the moments")
    p.add_argument("--think", action="store_true", help="let the model reason first (slower, sometimes better picks)")
    p.add_argument("--ollama", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    p.add_argument("--whisper", default="large-v3-turbo", help="faster-whisper model")
    p.add_argument("--language", help="spoken language code, e.g. en (default: detect)")
    p.add_argument("--captions", choices=["bold", "green", "clean", "none"], default="bold")
    p.add_argument("--no-hook", action="store_true", help="no title card at the start")
    p.add_argument("--hook-seconds", type=float, default=3.0)
    p.add_argument("--max-gap", type=float, default=0.5,
                   help="cut pauses longer than this many seconds (0 = keep all pauses)")
    p.add_argument("--layout", choices=["auto", "track", "fit", "center"], default="auto",
                   help="auto: follow faces, or fit with blurred background when there are none")
    p.add_argument("--encoder", choices=["auto", "nvenc", "x264"], default="auto")
    p.add_argument("--dry-run", action="store_true", help="pick the moments and stop before rendering")
    p.add_argument("--reselect", action="store_true", help="ask the AI again instead of reusing its last picks")
    p.add_argument("--retranscribe", action="store_true")
    args = p.parse_args()

    video = args.video
    if not video.exists():
        sys.exit(f"{video}: not found")
    if not FACE_MODEL.exists():
        sys.exit("Run ./setup.sh first (downloads the face model and caption font).")
    info = probe(video)
    out_dir = args.out or Path("clips") / slug(video.stem)
    work = out_dir / ".work"
    work.mkdir(parents=True, exist_ok=True)
    log(f"{video.name}: {info['w']}x{info['h']} @ {info['fps']:.2f} fps, {fmt_ts(info['duration'])} long")
    t_start = time.time()

    tr = transcribe(video, work, args.whisper, args.language, args.retranscribe)
    words = tr["words"]
    if not words and not args.clips:
        sys.exit("No speech found in this video (use --clips to cut it by hand).")
    args.spoken = LANGS.get(tr["language"], tr["language"])
    sents = sentences_from_words(words)
    (work / "transcript.txt").write_text("\n".join(f"[{i}] ({fmt_ts(s['s'])}) {s['text']}"
                                                   for i, s in enumerate(sents)))

    if args.clips:
        picks = []
        for k, rng in enumerate(args.clips.split(",")):
            a, b = (parse_ts(x) for x in rng.split("-"))
            b = min(b, info["duration"])
            ws = [i for i, w in enumerate(words) if w["s"] >= a - 0.05 and w["e"] <= b + 0.05]
            text = " ".join(words[i]["w"] for i in ws)
            picks.append({"i0": ws[0] if ws else None, "i1": ws[-1] if ws else None, "start": a, "end": b,
                          "score": 0, "title": " ".join(text.split()[:7]) or f"Clip {k + 1}",
                          "caption": text[:150], "hashtags": [], "why": "picked by hand"})
    else:
        sel_key = hashlib.sha1(json.dumps([args.model, args.count, args.min, args.max, args.think,
                                           len(words)]).encode()).hexdigest()[:10]
        sel_cache = work / f"picks-{sel_key}.json"
        if sel_cache.exists() and not args.reselect:
            picks = json.loads(sel_cache.read_text())
            log(f"[2/4] using the AI's previous picks ({sel_cache.name}; --reselect to ask again)")
        else:
            chosen = pick_clips(sents, args)
            picks = [{"i0": sents[c["a"]]["i0"], "i1": sents[c["b"]]["i1"], **{k: c[k] for k in
                      ("score", "title", "caption", "hashtags", "why")}} for c in chosen]
            sel_cache.write_text(json.dumps(picks, indent=1))
    if not picks:
        sys.exit("No good moments found. Try a lower --min, a higher --max, or --clips by hand.")

    for c in picks:
        if "start" not in c:
            c["start"], c["end"] = clip_bounds(words, c["i0"], c["i1"], info["duration"])
    log(f"[2/4] {len(picks)} moments:")
    for k, c in enumerate(picks, 1):
        log(f"   {k}. [{c['score']:>3}] {fmt_ts(c['start'])}-{fmt_ts(c['end'])} ({c['end'] - c['start']:.0f}s)  {c['title']}")
    if args.dry_run:
        return

    encoder = args.encoder
    if encoder == "auto":
        encoder = "nvenc" if nvenc_ok() else "x264"
    results = []
    for k, c in enumerate(picks, 1):
        name = f"{k:02d}_{slug(c['title'])}.mp4"
        log(f"[3/4] rendering {k}/{len(picks)}: {name}")
        try:
            r = render(video, info, c, words, out_dir / name, work, args, encoder)
        except RuntimeError as e:
            log(str(e))
            continue
        log(f"       {r['length']:.0f}s, {r['layout']}, trimmed {r['cut_s']:.1f}s of pauses, {r['render_s']:.0f}s to render")
        c.update(r, file=name)
        (out_dir / name).with_suffix(".txt").write_text(
            f"{c['title']}\n\n{c['caption']}\n\n" + " ".join("#" + h for h in c["hashtags"]) + "\n")
        results.append(c)

    (out_dir / "clips.json").write_text(json.dumps(
        [{k: c[k] for k in ("file", "score", "title", "caption", "hashtags", "why", "start", "end", "length")}
         for c in results], indent=1))
    write_gallery(out_dir, video.name, results)
    log(f"[4/4] done in {time.time() - t_start:.0f}s: {len(results)} shorts in {out_dir}/ "
        f"(open {out_dir / 'index.html'})")


if __name__ == "__main__":
    main()
