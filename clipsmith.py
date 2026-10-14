#!/usr/bin/env -S uv run --project /home/jt/claude/clipsmith python
"""clipsmith: our own Opus Clip. Turns a long video (file or URL) into ready-to-post shorts.

video -> Whisper (word timestamps) -> local LLM picks viral moments, scores them and writes titles,
captions and hashtags -> reframe to 9:16 / 4:5 / 1:1 following whoever is talking (or split screen for
two people) -> filler words and dead air cut, punch-in zooms, animated captions with keyword emphasis,
hook title, progress bar, logo, ducked background music, -14 LUFS -> MP4 + SRT + caption per clip.

GPU only: Whisper runs on CUDA and the picker model must sit fully in VRAM, or clipsmith stops and says
what to free. It never falls back to the CPU. `./serve.sh` starts the web app (web.py) on top of this.
"""
import argparse
import dataclasses
import gc
import hashlib
import html
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
FACE_MODEL = ASSETS / "face_detection_yunet_2023mar.onnx"
DEFAULT_FONT = (ASSETS / "Montserrat-ExtraBold.ttf", "Montserrat Thin ExtraBold")  # (file, full name)
ASPECTS = {"9:16": (1080, 1920), "4:5": (1080, 1350), "1:1": (1080, 1080)}
SAMPLE_FPS = 6.0
LANGS = {"en": "English", "pt": "Portuguese", "es": "Spanish", "fr": "French", "de": "German",
         "it": "Italian", "nl": "Dutch", "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}
FILLERS = {"um", "uh", "uhm", "umm", "uhh", "erm", "ah", "ahh", "hmm", "hm", "mm", "mmm", "hã", "ahn", "ãh",
           "ehh", "éé", "hum", "eeh"}
FILLERS_BY_LANG = {"en": {"er"}, "pt": {"eh"}, "es": {"eh", "em"}}


class GpuError(SystemExit):
    pass


# ---------------------------------------------------------------- reporting (CLI prints; the web app hooks in)

_hooks = []


def log(msg, progress=None):
    print(msg, file=sys.stderr, flush=True)
    for h in _hooks:
        h(msg, progress)


def fmt_ts(t):
    m, s = divmod(int(t), 60)
    return f"{m}:{s:02d}"


def parse_ts(s):
    """'83', '1:23', '1:02:03' or '83.5' -> seconds."""
    t = 0.0
    for p in str(s).strip().split(":"):
        t = t * 60 + float(p)
    return t


def slug(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "clip"


def hex_to_ass(h):
    h = h.lstrip("#")
    return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def norm_word(w):
    return re.sub(r"[^\w]", "", w.lower())


# ---------------------------------------------------------------- settings

@dataclasses.dataclass
class Settings:
    count: int = 5
    min: float = 20
    max: float = 60
    prompt: str = ""                 # "find moments about pricing"
    model: str = "qwen3.6:27b"
    think: bool = False
    ollama: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    whisper: str = "large-v3-turbo"
    language: str = ""
    aspect: str = "9:16"
    layout: str = "auto"             # auto | track | split | fit | center
    captions: str = "bold"           # bold | green | karaoke | pop | clean | none
    caption_pos: str = "auto"        # auto | low | middle
    highlight: str = "#FFE500"       # active word
    emphasis: str = "#39FF14"        # AI-picked keywords
    hook: bool = True
    hook_seconds: float = 3.0
    progress_bar: bool = True
    zoom: bool = True                # punch-ins on cuts and keywords
    fillers: bool = True             # cut um/uh
    max_gap: float = 0.5             # cut pauses longer than this (0 = keep)
    logo: str = ""
    logo_pos: str = "top-right"
    logo_width: int = 170
    music: str = ""                  # file or folder
    music_volume: float = 0.12
    font_file: str = ""
    encoder: str = "auto"
    maxrate: str = "16M"

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def to_dict(self):
        return dataclasses.asdict(self)


def load_brand(path):
    """brand.json: any Settings fields (fonts, colors, logo, music...) to use as defaults."""
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------- input

def fetch_url(url, dest):
    import yt_dlp
    dest.mkdir(parents=True, exist_ok=True)
    log(f"[0/4] downloading {url} ...")
    opts = {"format": "bv*[height<=2160][ext=mp4]+ba[ext=m4a]/bv*[height<=2160]+ba/b", "merge_output_format": "mp4",
            "outtmpl": str(dest / "%(title).80s [%(id)s].%(ext)s"), "quiet": True, "noprogress": True,
            "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=True)
        path = Path(y.prepare_filename(info)).with_suffix(".mp4")
    if not path.exists():
        cands = sorted(dest.glob(f"*[[]{info['id']}[]]*"), key=lambda p: p.stat().st_mtime)
        path = cands[-1]
    log(f"[0/4] downloaded {path.name}")
    return path


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
                          "-show_format", str(path)], capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise SystemExit(f"{path}: no video stream")
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


# ---------------------------------------------------------------- transcription (GPU only)

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


def transcribe(video, work, model_name, language, force=False):
    st = video.stat()
    key = hashlib.sha1(f"{video.resolve()}|{st.st_size}|{st.st_mtime}|{model_name}|{language or None}".encode()).hexdigest()[:12]
    cache = work / f"transcript-{key}.json"
    if cache.exists() and not force:
        log(f"[1/4] transcript: cached ({cache.name})", 0.25)
        return {**json.loads(cache.read_text()), "file": cache.name}

    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(pcm, np.float32)
    load_cuda12_libs()
    from faster_whisper import BatchedInferencePipeline, WhisperModel
    t0 = time.time()
    # GPU only, never the CPU. The GPU is shared (Ollama, ComfyUI...), so retry with a smaller batch.
    words, tinfo = None, None
    for batch in (16, 4):
        model = pipe = None
        try:
            model = WhisperModel(model_name, device="cuda", compute_type="float16")
            pipe = BatchedInferencePipeline(model=model)
            log(f"[1/4] transcribing with Whisper {model_name} on the GPU...", 0.05)
            segments, tinfo = pipe.transcribe(audio, batch_size=batch, word_timestamps=True, vad_filter=True,
                                              language=language or None)
            words = [{"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)}
                     for seg in segments for w in (seg.words or []) if w.word.strip()]
            break
        except Exception as e:  # noqa: BLE001
            if batch == 4:
                raise GpuError(f"Whisper can't run on the GPU ({str(e)[:100]}). Free some VRAM (check "
                               "`nvidia-smi` and `ollama ps`) and try again. clipsmith never uses the CPU.")
            log(f"[1/4] GPU attempt failed ({str(e)[:80]}); retrying with a smaller batch")
        finally:
            pipe = model = None
            gc.collect()
    data = {"language": tinfo.language, "words": words}
    cache.write_text(json.dumps(data))
    log(f"[1/4] transcribed {len(words)} words in {time.time() - t0:.0f}s (language: {tinfo.language})", 0.25)
    return {**data, "file": cache.name}


def sentences_from_words(words):
    """Group words into sentence-ish lines: split on . ? ! or a long pause, and keep lines short."""
    out, cur = [], []
    for i, w in enumerate(words):
        cur.append(i)
        nxt = words[i + 1] if i + 1 < len(words) else None
        gap = (nxt["s"] - w["e"]) if nxt else 99
        if w["w"][-1] in ".?!" or gap > 0.9 or (len(cur) >= 30 and (w["w"][-1] in ",;:" or gap > 0.3)) or len(cur) >= 45:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [{"i0": c[0], "i1": c[-1], "s": words[c[0]]["s"], "e": words[c[-1]]["e"],
             "text": " ".join(words[j]["w"] for j in c)} for c in out]


# ---------------------------------------------------------------- picking moments (the LLM, GPU only)

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
{focus}
Rules:
- Each clip spans whole lines: start_line to end_line (inclusive), using the line numbers shown.
- Each clip lasts between {min_s} and {max_s} seconds (use the timestamps). Clips must not overlap.
- Return up to {n} clips, best first. Fewer is fine if the material is weak; never pad with filler.
- Scores are honest 0-100 estimates: hook (first 3 seconds), flow (coherent, complete), value \
(insight, emotion or entertainment), trend (fits what performs on Reels now), and score = overall viral \
potential (90+ only for exceptional moments).
- title: on-screen hook text for the first seconds, max 8 words, punchy, no hashtags, no emojis.
- caption: Instagram caption, 1-2 short sentences that tease the clip, plus a light call to action.
- hashtags: 5-8 relevant hashtags, without the # sign.
- keywords: 3-6 single words, copied exactly as spoken in the clip, that carry the punch (numbers, \
strong nouns, emotional words). They get highlighted in the captions.
- why: one sentence on why this moment works.
- Write title, caption, hashtags in the same language as the transcript ({language}), in the speaker's \
own voice and vocabulary. Write "why" in English."""

CLIP_SCHEMA = {
    "type": "object",
    "properties": {"clips": {"type": "array", "items": {
        "type": "object",
        "properties": {"start_line": {"type": "integer"}, "end_line": {"type": "integer"},
                       "hook": {"type": "integer"}, "flow": {"type": "integer"}, "value": {"type": "integer"},
                       "trend": {"type": "integer"}, "score": {"type": "integer"}, "title": {"type": "string"},
                       "caption": {"type": "string"}, "hashtags": {"type": "array", "items": {"type": "string"}},
                       "keywords": {"type": "array", "items": {"type": "string"}}, "why": {"type": "string"}},
        "required": ["start_line", "end_line", "hook", "flow", "value", "trend", "score", "title", "caption",
                     "hashtags", "keywords", "why"]}}},
    "required": ["clips"],
}


def _post(url, body, timeout):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ensure_model_on_gpu(url, model, num_ctx):
    """Load the picker model and make sure Ollama put all of it in VRAM (not partly on the CPU)."""
    try:
        _post(url + "/api/generate", {"model": model, "prompt": "", "keep_alive": "5m",
                                      "options": {"num_ctx": num_ctx}}, 600)
        ps = json.loads(urllib.request.urlopen(url + "/api/ps", timeout=10).read())
    except urllib.error.URLError as e:
        raise SystemExit(f"Can't reach Ollama at {url} ({getattr(e, 'reason', e)}). Is `ollama serve` running?")
    m = next((m for m in ps.get("models", []) if m["name"] == model or m["model"] == model), None)
    if m and m.get("size_vram", 0) < m.get("size", 0) * 0.99:
        ollama_unload(url, model)
        others = ", ".join(x["name"] for x in ps["models"] if x["name"] != model) or "nothing else"
        raise GpuError(f"{model} doesn't fit in VRAM (only {m['size_vram'] / 1e9:.1f} of {m['size'] / 1e9:.1f} GB "
                       f"on the GPU; also loaded: {others}). Free VRAM or pick a smaller --model. "
                       "clipsmith never runs models on the CPU.")


def ollama_chat(url, model, system, user, think, num_ctx):
    body = {"model": model, "stream": False, "format": CLIP_SCHEMA, "think": think, "keep_alive": "5m",
            "options": {"temperature": 0.4, "num_ctx": num_ctx},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    for attempt in range(2):
        try:
            return _post(url + "/api/chat", body, 1800)["message"]["content"]
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
        _post(url + "/api/generate", {"model": model, "keep_alive": 0}, 30)
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


def pick_clips(sents, s: Settings, language):
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

    url = s.ollama.rstrip("/")
    # context sized to the longest window (about 3 chars per token, plus prompt and answer), so the model
    # and its cache fit in VRAM
    longest = max(sum(len(sents[i]["text"]) + 12 for i in range(a, b + 1)) for a, b in windows)
    num_ctx = min(32768, max(8192, math.ceil((longest / 3 + 4000) / 2048) * 2048))
    ensure_model_on_gpu(url, s.model, num_ctx)
    per_window = max(3, math.ceil(s.count * 1.5 / len(windows)) + 1)
    focus = (f"\nThe creator wants clips about: {s.prompt}\nOnly pick moments that match this; return fewer "
             "clips rather than off-topic ones.\n") if s.prompt else ""
    system = EDITOR_PROMPT.format(min_s=s.min, max_s=s.max, n=per_window, language=language, focus=focus)
    cands = []
    for wi, (a, b) in enumerate(windows):
        lines = "\n".join(f"[{i}] ({fmt_ts(sents[i]['s'])}) {sents[i]['text']}" for i in range(a, b + 1))
        t0 = time.time()
        log(f"[2/4] {s.model} reading transcript part {wi + 1}/{len(windows)} "
            f"({fmt_ts(sents[a]['s'])}-{fmt_ts(sents[b]['e'])})...", 0.25 + 0.15 * wi / len(windows))
        raw = ollama_chat(url, s.model, system, f"Transcript:\n{lines}", s.think, num_ctx)
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
            clamp = lambda v: max(0, min(100, int(v)))
            c = fit_length({"a": lo, "b": hi, "score": clamp(c["score"]),
                            "scores": {k: clamp(c[k]) for k in ("hook", "flow", "value", "trend")},
                            "title": c["title"].strip().strip('"'), "caption": c["caption"].strip(),
                            "hashtags": [h.strip().lstrip("#").replace(" ", "") for h in c["hashtags"] if h.strip()],
                            "keywords": [k.strip() for k in c["keywords"] if k.strip()][:6],
                            "why": c["why"].strip()}, sents, s.min, s.max)
            if c:
                cands.append(c)
    ollama_unload(url, s.model)   # hand the GPU back

    chosen = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        a, b = sents[c["a"]]["s"], sents[c["b"]]["e"]
        if all(min(b, sents[o["b"]]["e"]) - max(a, sents[o["a"]]["s"]) < 0.25 * (b - a) for o in chosen):
            chosen.append(c)
        if len(chosen) >= s.count:
            break
    return chosen


# ---------------------------------------------------------------- clip timing, fillers and dead air

def clip_bounds(words, i0, i1, duration):
    s = words[i0]["s"] - 0.15
    if i0 > 0:
        s = max(s, words[i0 - 1]["e"] + 0.02)
    e = words[i1]["e"] + 0.4
    if i1 + 1 < len(words):
        e = min(e, max(words[i1]["e"] + 0.1, words[i1 + 1]["s"] - 0.05))
    return max(0.0, s), min(duration, e)


def words_in(words, start, end):
    """Indices of the words inside [start, end] (for hand-made or edited clips)."""
    ws = [i for i, w in enumerate(words) if w["s"] >= start - 0.05 and w["e"] <= end + 0.05]
    return (ws[0], ws[-1]) if ws else (None, None)


def is_filler(w, language):
    n = norm_word(w)
    return n in FILLERS or n in FILLERS_BY_LANG.get(language, set())


def keep_intervals(words, i0, i1, cs, ce, max_gap, cut_fillers, language):
    """Pieces of [cs, ce] to keep (clip-relative seconds): cuts pauses longer than max_gap and filler words.
    Returns (pieces, kept word indices)."""
    idx = list(range(i0, i1 + 1))
    kept = [i for i in idx if not (cut_fillers and is_filler(words[i]["w"], language))] or idx
    pieces, a = [], 0.0
    for p, q in zip(kept, kept[1:]):
        gap = words[q]["s"] - words[p]["e"]
        skipped = q - p > 1     # a filler sits between them
        if (max_gap > 0 and gap > max_gap) or (skipped and gap > 0.15):
            pieces.append((a, words[p]["e"] - cs + (0.12 if not skipped else 0.06)))
            a = words[q]["s"] - cs - (0.1 if not skipped else 0.05)
    pieces.append((a, ce - cs))
    pieces = [(max(0.0, p), min(ce - cs, q)) for p, q in pieces if q - p > 0.05]
    return pieces, kept


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

    def piece_at(self, t):
        for k, (a, b) in enumerate(self.pieces):
            if t <= b:
                return k
        return len(self.pieces) - 1


# ---------------------------------------------------------------- faces: who is where, who is talking

def analyze_faces(video, info, cs, dur):
    """Sample the clip, find faces, and estimate who is talking from mouth motion.
    Returns (samples, aw, ah) where samples = [(t, [(cx, cy, w, h, mouth_motion)])] in analysis pixels."""
    import cv2
    aw = 640
    ah = int(round(aw * info["h"] / info["w"] / 2)) * 2
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-hwaccel", "cuda", "-ss", f"{cs:.3f}", "-t", f"{dur:.3f}",
                             "-i", str(video), "-vf", f"fps={SAMPLE_FPS},scale={aw}:{ah}", "-f", "rawvideo",
                             "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    det = cv2.FaceDetectorYN.create(str(FACE_MODEL), "", (aw, ah), 0.7, 0.3, 50)
    samples, prev, n = [], None, 0
    fsize = aw * ah * 3
    box = lambda x0, y0, x1, y1: (slice(max(0, int(y0)), min(ah, int(y1))), slice(max(0, int(x0)), min(aw, int(x1))))
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
            mouth = box(mx0 - 0.12 * w, my - 0.12 * h, mx1 + 0.12 * w, my + 0.22 * h)
            face = box(x, y, x + w, y + h)
            motion = 0.0
            if prev is not None and gray[mouth].size and gray[face].size:
                m = np.abs(gray[mouth] - prev[mouth]).mean()
                head = np.abs(gray[face] - prev[face]).mean()
                motion = max(0.0, m - 0.7 * head)
            found.append((x + w / 2, y + h / 2, w, h, motion))
        samples.append((n / SAMPLE_FPS, found))
        prev, n = gray, n + 1
    proc.wait()
    return samples, aw, ah


def track_faces(samples):
    """Link detections over time into tracks. Returns per-sample [(track_id, cx, cy, h, score)] and
    per-track stats {id: {"n": samples seen, "cx": [...], "cy": [...], "h": [...]}}."""
    tracks, nid, per_sample, stats = [], 0, [], {}
    for t, faces in samples:
        here = []
        for cx, cy, w, h, motion in faces:
            best = None
            for tr in tracks:
                d = math.hypot(tr["cx"] - cx, tr["cy"] - cy)
                if t - tr["last"] < 1.5 and d < 0.6 * max(w, tr["w"]) and (best is None or d < best[0]):
                    best = (d, tr)
            if best is None:
                tr = {"id": nid, "cx": cx, "cy": cy, "w": w, "h": h, "act": 0.0, "last": t}
                nid += 1
                tracks.append(tr)
            else:
                tr = best[1]
            tr.update(cx=cx, cy=cy, w=w, h=h, last=t, act=0.75 * tr["act"] + 0.25 * motion)
            here.append(tr)
            st = stats.setdefault(tr["id"], {"n": 0, "cx": [], "cy": [], "h": []})
            st["n"] += 1
            st["cx"].append(cx)
            st["cy"].append(cy)
            st["h"].append(h)
        top = max((tr["act"] for tr in here), default=0) or 1.0
        per_sample.append([(tr["id"], tr["cx"], tr["cy"], tr["h"], tr["h"] * (0.3 + tr["act"] / top)) for tr in here])
    return per_sample, stats


def plan_camera(per_sample, aw, ah, crop_w_a):
    """Where the frame looks over time: follows the active speaker, hard-cuts when the speaker changes
    (like a real editor) and glides smoothly otherwise. Returns dict of per-sample arrays + cut indices."""
    cur, challenger, streak, missing = None, None, 0, 0
    hold, patience = int(1.0 * SAMPLE_FPS), int(0.7 * SAMPLE_FPS)
    xs, ys, hs, ids = [], [], [], []
    for faces in per_sample:
        by_id = {f[0]: f for f in faces}
        best = max(faces, key=lambda f: f[4]) if faces else None
        if cur not in by_id:
            missing += 1
            if best and (cur is None or missing >= patience):
                cur, missing, challenger, streak = best[0], 0, None, 0
        else:
            missing = 0
            if best[0] != cur and best[4] > 1.3 * by_id[cur][4]:
                streak = streak + 1 if challenger == best[0] else 1
                challenger = best[0]
                if streak >= hold:
                    cur, challenger, streak = best[0], None, 0
            else:
                challenger, streak = None, 0
        f = by_id.get(cur)
        xs.append(f[1] if f else None)
        ys.append(f[2] if f else None)
        hs.append(f[3] if f else None)
        ids.append(cur)

    def fill(v, default):
        first = next((x for x in v if x is not None), default)
        out = []
        for x in v:
            out.append(x if x is not None else (out[-1] if out else first))
        return np.array(out, float)

    xs, ys, hs = fill(xs, aw / 2), fill(ys, ah * 0.4), fill(hs, ah * 0.3)
    first_id = next((t for t in ids if t is not None), None)
    ids = [first_id if t is None else t for t in ids]
    cuts = [i for i in range(1, len(ids)) if ids[i] != ids[i - 1]]
    bounds = [0] + cuts + [len(xs)]
    dz = 0.08 * crop_w_a          # deadzone: small head moves don't move the camera
    win = max(1, int(0.8 * SAMPLE_FPS))
    for arr, dead in ((xs, dz), (ys, 0.05 * ah), (hs, 0.0)):
        for a, b in zip(bounds, bounds[1:]):
            seg = arr[a:b].copy()
            pos = seg[0]
            for k in range(len(seg)):
                d = seg[k] - pos
                if abs(d) > dead:
                    pos += d - math.copysign(dead, d)
                seg[k] = pos
            if len(seg) > 2:      # gentle zero-phase smoothing
                pad = np.pad(seg, win, mode="edge")
                seg = np.convolve(pad, np.ones(2 * win + 1) / (2 * win + 1), mode="same")[win:-win]
            arr[a:b] = seg
    return {"x": xs, "y": ys, "h": hs, "cuts": cuts}


def detect_split(per_sample, stats, aw, crop_w_a):
    """Two people who are both on screen most of the time and too far apart for one vertical frame ->
    the two track ids (left, right), else None."""
    n = len(per_sample)
    big = [tid for tid, st in stats.items() if st["n"] >= 0.5 * n]
    if len(big) < 2:
        return None
    big.sort(key=lambda t: -stats[t]["n"])
    a, b = big[:2]
    ax, bx = np.median(stats[a]["cx"]), np.median(stats[b]["cx"])
    if abs(ax - bx) < 0.9 * crop_w_a:
        return None
    return (a, b) if ax < bx else (b, a)


# ---------------------------------------------------------------- zoom plan

def zoom_curve(tl, kw_times, enabled):
    """z(t) for clip-relative source time: alternate punch-in on every jump cut (hides the cut, keeps
    energy up) plus a quick pop on AI-picked keywords."""
    if not enabled:
        return lambda t: 1.0
    pops, last = [], -99.0
    for s, e in kw_times:
        if s - last >= 4.0:
            pops.append((s, e))
            last = s

    def z(t):
        base = 1.0 + 0.08 * (tl.piece_at(t) % 2)
        pop = 0.0
        for s, e in pops:
            if s - 0.12 <= t <= e + 0.75:
                if t < s:
                    k = (t - (s - 0.12)) / 0.12
                elif t <= e + 0.45:
                    k = 1.0
                else:
                    k = 1 - (t - e - 0.45) / 0.3
                pop = max(pop, 0.1 * (k * k * (3 - 2 * k)))   # smoothstep
        return min(1.3, base + pop)
    return z


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


CAPTION_STYLES = {
    #  size, words per caption, uppercase, how the spoken word is shown
    "bold": dict(size=100, words=3, upper=True, mode="highlight"),
    "green": dict(size=100, words=3, upper=True, mode="highlight", swap=True),
    "karaoke": dict(size=92, words=4, upper=True, mode="karaoke"),
    "pop": dict(size=135, words=1, upper=True, mode="pop"),
    "clean": dict(size=70, words=5, upper=False, mode="plain"),
}


def caption_chunks(ws, per):
    chunks, cur = [], []
    for i, w in enumerate(ws):
        cur.append(w)
        nxt = ws[i + 1] if i + 1 < len(ws) else None
        chars = sum(len(x["w"]) + 1 for x in cur)
        if (not nxt or len(cur) >= per or chars + len(nxt["w"]) > 18 + 6 * (per > 3)
                or w["w"][-1] in ".?!,;:" or nxt["s"] - w["e"] > 0.35):
            chunks.append(cur)
            cur = []
    return chunks


def write_ass(path, ws, length, title, s: Settings, ow, oh, font_name, hook_y, cap_y):
    """ws: caption words already on the output timeline, each {w, s, e, kw}."""
    st = CAPTION_STYLES.get(s.captions)
    hi, em = hex_to_ass(s.highlight), hex_to_ass(s.emphasis)
    if st and st.get("swap"):
        hi, em = em, hi
    size = st["size"] if st else 80
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {ow}
PlayResY: {oh}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font_name},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,1,0,1,7,3,2,60,60,{oh - cap_y},1
Style: Hook,{font_name},84,&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,18,0,8,90,90,0,1
Style: Bar,Arial,10,{hi},{hi},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    if s.progress_bar and length > 1:
        bh = 12
        ev.append(f"Dialogue: 2,{ass_time(0)},{ass_time(length)},Bar,,0,0,0,,{{\\an7\\pos(0,0)\\clip(0,0,0,{bh})"
                  f"\\t(0,{int(length * 1000)},\\clip(0,0,{ow},{bh}))\\p1}}m 0 0 l {ow} 0 {ow} {bh} 0 {bh}{{\\p0}}")
    if title and s.hook and s.hook_seconds > 0:
        txt = "\\N".join(ass_text(l) for l in wrap(title, 20))
        ev.append(f"Dialogue: 1,{ass_time(0)},{ass_time(min(s.hook_seconds, length))},Hook,,0,0,0,,"
                  f"{{\\an8\\pos({ow // 2},{hook_y})\\fad(150,250)\\fscx90\\fscy90\\t(0,180,\\fscx100\\fscy100)}}{txt}")
    if st:
        def word(x, active=False, past=False):
            t = ass_text(x["w"].upper() if st["upper"] else x["w"])
            if st["mode"] == "plain":
                return t
            color = hi if active or (past and st["mode"] == "karaoke") else (em if x["kw"] else None)
            grow = 112 if active and st["mode"] != "pop" else (115 if x["kw"] and st["mode"] != "pop" else 100)
            if color is None and grow == 100:
                return t
            c = f"\\c{color}" if color else ""
            return f"{{{c}\\fscx{grow}\\fscy{grow}}}{t}{{\\c&H00FFFFFF&\\fscx100\\fscy100}}"

        chunks = caption_chunks(ws, st["words"])
        for ci, ch in enumerate(chunks):
            end = min(ch[-1]["e"] + 0.5, chunks[ci + 1][0]["s"]) if ci + 1 < len(chunks) else min(length, ch[-1]["e"] + 0.5)
            for k, w in enumerate(ch):
                a = w["s"]
                b = ch[k + 1]["s"] if k + 1 < len(ch) else end
                if b - a < 0.02:
                    continue
                if st["mode"] == "pop":
                    color = f"\\c{em}" if w["kw"] else ""
                    text = f"{{{color}\\fscx70\\fscy70\\t(0,90,\\fscx108\\fscy108)\\t(90,160,\\fscx100\\fscy100)}}" + \
                        ass_text(w["w"].upper())
                else:
                    parts = [word(x, active=(j == k), past=(j < k)) for j, x in enumerate(ch)]
                    pop = "{\\fscx85\\fscy85\\t(0,90,\\fscx100\\fscy100)}" if k == 0 else ""
                    text = pop + " ".join(parts)
                ev.append(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Cap,,0,0,0,,{text}")
    Path(path).write_text(head + "\n".join(ev) + "\n", encoding="utf-8")


def write_srt(path, ws):
    def ts(t):
        ms = int(round(max(0.0, t) * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    out, line, n = [], [], 0
    for i, w in enumerate(ws):
        line.append(w)
        nxt = ws[i + 1] if i + 1 < len(ws) else None
        if not nxt or len(" ".join(x["w"] for x in line)) > 34 or w["w"][-1] in ".?!" or nxt["s"] - w["e"] > 0.6:
            n += 1
            end = min(line[-1]["e"] + 0.1, nxt["s"]) if nxt else line[-1]["e"] + 0.1
            out.append(f"{n}\n{ts(line[0]['s'])} --> {ts(end)}\n{' '.join(x['w'] for x in line)}\n")
            line = []
    Path(path).write_text("\n".join(out), encoding="utf-8")


# ---------------------------------------------------------------- rendering

def fesc(p):
    """Escape a path for use inside an ffmpeg filtergraph option."""
    return str(p).replace("\\", "\\\\\\\\").replace(":", "\\\\:").replace("'", "\\\\\\'").replace(",", "\\,")


_nvenc = None


def nvenc_ok():
    global _nvenc
    if _nvenc is None:
        r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
                            "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True)
        _nvenc = r.returncode == 0
    return _nvenc


def font_setup(work, s: Settings):
    """A fonts folder for libass with our caption font (and the brand font, if any)."""
    fonts = work / "fonts"
    fonts.mkdir(exist_ok=True)
    src, name = DEFAULT_FONT
    if s.font_file and Path(s.font_file).exists():
        src = Path(s.font_file)
        out = subprocess.run(["fc-scan", "--format", "%{fullname}", str(src)], capture_output=True, text=True).stdout
        name = out.split(",")[0].strip() or src.stem
    link = fonts / src.name
    if not link.exists():
        link.symlink_to(src.resolve())
    return fonts, name


def pick_music(s: Settings, k):
    if not s.music:
        return None
    p = Path(s.music).expanduser()
    if p.is_dir():
        files = sorted(x for x in p.iterdir() if x.suffix.lower() in (".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac"))
        return random.Random(k).choice(files) if files else None
    return p if p.exists() else None


def render_clip(video, info, clip, words, language, out_path, work, s: Settings, k=0):
    ow, oh = ASPECTS.get(s.aspect, ASPECTS["9:16"])
    cs, ce = clip["start"], clip["end"]
    dur = ce - cs
    spoken = clip.get("i0") is not None
    if spoken:
        pieces, kept = keep_intervals(words, clip["i0"], clip["i1"], cs, ce, s.max_gap, s.fillers, language)
    else:
        pieces, kept = [(0.0, dur)], []
    tl = Timeline(pieces)
    fps = min(info["fps"], 60.0)
    tag = out_path.stem
    W, H = info["w"], info["h"]
    a_out = ow / oh

    # caption words on the output timeline, with keyword flags
    kws = {norm_word(x) for x in clip.get("keywords", [])} - {""}
    ws = [{"w": words[i]["w"], "s": tl(words[i]["s"] - cs), "e": tl(words[i]["e"] - cs),
           "kw": norm_word(words[i]["w"]) in kws} for i in kept]
    kw_times = [(words[i]["s"] - cs, words[i]["e"] - cs) for i in kept if norm_word(words[i]["w"]) in kws]

    # base window: the largest output-shaped rectangle that fits in the source
    if W / H > a_out:
        bw, bh = H * a_out, float(H)
    else:
        bw, bh = float(W), W / a_out
    layout = s.layout
    samples = per_sample = cam = split = None
    aw, ah = 640, int(round(640 * H / W / 2)) * 2
    if layout in ("auto", "track", "split", "center"):
        samples, aw, ah = analyze_faces(video, info, cs, dur)
        per_sample, stats = track_faces(samples)
        faced = sum(1 for f in per_sample if f) / max(1, len(per_sample))
        sc = aw / W
        if layout in ("auto", "split") and W / H > a_out * 1.2:
            split = detect_split(per_sample, stats, aw, bw * sc)
            if split:
                layout = "split"
        if layout != "split":
            if layout == "auto":
                layout = "track" if faced >= 0.35 else "fit"
            if layout in ("track", "center"):
                cam = plan_camera(per_sample, aw, ah, bw * sc) if faced > 0 else None
    elif layout == "split":
        layout = "track"

    zf = zoom_curve(tl, kw_times, s.zoom and layout in ("track", "center"))
    cap_y = int(oh * (0.71 if s.aspect == "9:16" else 0.8))
    if s.caption_pos == "middle":
        cap_y = int(oh * 0.58)
    hook_y = int(oh * 0.13)
    inputs = ["-hwaccel", "cuda", "-ss", f"{cs:.3f}", "-t", f"{dur:.3f}", "-i", str(Path(video).resolve())]

    if layout == "split":
        sc = W / aw
        halves = []
        for tid in split:
            st = stats[tid]
            fx, fy, fh = (float(np.median(st[k])) * sc for k in ("cx", "cy", "h"))
            ch = min(float(H), 3.4 * fh)
            cw = ch * ow / (oh / 2)
            if cw > W:
                cw, ch = float(W), W * (oh / 2) / ow
            x = min(max(fx - cw / 2, 0), W - cw)
            y = min(max(fy - 0.42 * ch, 0), H - ch)
            halves.append(f"crop={int(cw) // 2 * 2}:{int(ch) // 2 * 2}:{int(x)}:{int(y)},"
                          f"scale={ow}:{oh // 2}:flags=lanczos,setsar=1")
        vchain = (f"[0:v]split[ta][tb];[ta]{halves[0]}[top];[tb]{halves[1]}[bot];"
                  f"[top][bot]vstack,fps={fps:.4f}[v0]")
        how = "split screen (two speakers)"
        cap_y = oh // 2 + 150 if s.caption_pos != "low" else cap_y
        hook_y = oh // 2 - 330
    elif layout in ("track", "center"):
        sc = W / aw
        lines, last = [], None
        times = np.array([t for t, _ in samples]) if samples else np.array([0.0])
        cut_times = [times[i] for i in cam["cuts"]] if cam else []
        frames = int(dur * fps) + 2
        first = None
        for f in range(frames):
            t = f / fps
            if cam:
                seg = sum(1 for c in cut_times if t >= c)
                lo = 0 if seg == 0 else cam["cuts"][seg - 1]
                hi = cam["cuts"][seg] if seg < len(cam["cuts"]) else len(times)
                fx = np.interp(t, times[lo:hi], cam["x"][lo:hi]) * sc
                fy = np.interp(t, times[lo:hi], cam["y"][lo:hi]) * sc
            else:
                fx, fy = W / 2, H * 0.4
            if layout == "center":
                fx = W / 2
            z = zf(t)
            w, h = bw / z, bh / z
            x = min(max(fx - w / 2, 0), W - w)
            y = min(max(fy - 0.38 * h, 0), H - h) if (z > 1.001 or bh < H) else (H - h) / 2
            v = (int(w) // 2 * 2, int(h) // 2 * 2, int(x) // 2 * 2, int(y) // 2 * 2)
            if first is None:
                first = v
            if v != last:
                cmds = []
                for name, val, old in zip("whxy", v, last or (None,) * 4):
                    if val != old:
                        cmds.append(f"crop@rf {name} {val}")
                lines.append(f"{t:.4f} " + ", ".join(cmds) + ";")
                last = v
        cmd_file = work / f"{tag}.cmd"
        cmd_file.write_text("\n".join(lines) + "\n")
        vchain = (f"[0:v]sendcmd=f={fesc(cmd_file.name)},crop@rf=w={first[0]}:h={first[1]}:x={first[2]}:y={first[3]},"
                  f"scale={ow}:{oh}:flags=lanczos,setsar=1,fps={fps:.4f}[v0]")
        how = ("speaker tracking" + (f", {len(cam['cuts'])} camera cuts" if cam and cam["cuts"] else "")) \
            if layout == "track" else "center crop"
        if s.zoom:
            how += ", zooms"
        # keep the hook title off the face: above the head if there's room, else under the chin
        if cam is not None:
            fy_top = np.median(cam["y"] - 0.75 * cam["h"]) * sc
            fy_bot = np.median(cam["y"] + 0.6 * cam["h"]) * sc
            y0 = min(max(np.median(cam["y"]) * sc - 0.38 * bh, 0), H - bh) if bh < H else 0
            top, bottom = (fy_top - y0) * oh / bh, (fy_bot - y0) * oh / bh
            height = len(wrap(clip.get("title", ""), 20)) * 105 + 40
            if top - 40 < hook_y + height and bottom + 30 + height <= cap_y - 130:
                hook_y = int(bottom + 30)
    else:   # fit: whole frame over a blurred copy of itself
        vchain = (f"[0:v]split[bgi][fgi];[bgi]scale={ow}:{oh}:force_original_aspect_ratio=increase,"
                  f"crop={ow}:{oh},gblur=sigma=40,eq=brightness=-0.12[bg];"
                  f"[fgi]scale={ow}:{oh}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
                  f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={fps:.4f}[v0]")
        how = "fit with blurred background"

    fonts, font_name = font_setup(work, s)
    ass = work / f"{tag}.ass"
    write_ass(ass, ws, tl.length, clip.get("title", ""), s, ow, oh, font_name, hook_y, cap_y)
    write_srt(out_path.with_suffix(".srt"), ws)

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
    g.append(f"[vc]subtitles=f={fesc(ass.name)}:fontsdir={fesc(fonts.resolve())}[vsub]")

    vlast, nin = "vsub", 1
    logo = Path(s.logo).expanduser() if s.logo else None
    if logo and logo.exists():
        inputs += ["-i", str(logo.resolve())]
        m = 40
        pos = {"top-left": f"{m}:{m + 30}", "top-right": f"W-w-{m}:{m + 30}",
               "bottom-left": f"{m}:H-h-{int(oh * 0.2)}", "bottom-right": f"W-w-{m}:H-h-{int(oh * 0.2)}"}
        g.append(f"[{nin}:v]scale={s.logo_width}:-1,format=rgba,colorchannelmixer=aa=0.9[lg]")
        g.append(f"[vsub][lg]overlay={pos.get(s.logo_pos, pos['top-right'])}[vlogo]")
        vlast, nin = "vlogo", nin + 1
    music = pick_music(s, k)
    if has_a:
        if music:
            inputs += ["-stream_loop", "-1", "-i", str(music.resolve())]
            L = tl.length
            g.append(f"[{nin}:a]atrim=0:{L:.3f},asetpts=PTS-STARTPTS,aresample=48000,volume={s.music_volume},"
                     f"afade=t=in:d=0.5,afade=t=out:st={max(0, L - 1.5):.3f}:d=1.5[mus]")
            g.append("[ac]aresample=48000,asplit[sp][key]")
            g.append("[mus][key]sidechaincompress=threshold=0.03:ratio=6:attack=15:release=350[duck]")
            g.append("[sp][duck]amix=inputs=2:duration=first:normalize=0[mix]")
            alast = "mix"
        else:
            alast = "ac"
        g.append(f"[{alast}]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]")
    graph = work / f"{tag}.filter"
    graph.write_text(";\n".join(g))

    enc = s.encoder if s.encoder != "auto" else ("nvenc" if nvenc_ok() else "x264")
    venc = (["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0",
             "-maxrate", s.maxrate, "-bufsize", "32M", "-spatial-aq", "1", "-temporal-aq", "1"]
            if enc == "nvenc" else ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-maxrate", s.maxrate,
                                    "-bufsize", "32M"])
    cmd = ["ffmpeg", "-v", "error", "-y"] + inputs + ["-filter_complex_script", graph.name, "-map", f"[{vlast}]"]
    if has_a:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    cmd += venc + ["-profile:v", "high", "-pix_fmt", "yuv420p", "-g", str(int(round(fps * 2))),
                   "-shortest", "-movflags", "+faststart", str(out_path.resolve())]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"ffmpeg failed for {tag}:\n{r.stderr[-3000:]}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{min(1.2, tl.length / 2):.2f}", "-i", str(out_path),
                    "-frames:v", "1", "-q:v", "3", str(out_path.with_suffix(".jpg"))], check=False)
    return {"layout": how, "length": round(tl.length, 2), "cut_s": round(dur - tl.length, 2),
            "render_s": round(time.time() - t0, 1), "aspect": s.aspect}


# ---------------------------------------------------------------- projects

class Project:
    """One source video, its transcript, the picked clips and how they were rendered (project.json)."""

    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.work = self.dir / ".work"
        self.file = self.dir / "project.json"
        self.data = json.loads(self.file.read_text()) if self.file.exists() else {}
        self._words = None

    @property
    def clips(self):
        return self.data.setdefault("clips", [])

    @property
    def settings(self):
        return Settings.from_dict(self.data.get("settings"))

    @property
    def words(self):
        if self._words is None:
            self._words = json.loads((self.work / self.data["transcript"]).read_text())["words"]
        return self._words

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))
        tmp.replace(self.file)

    def clip(self, cid):
        return next(c for c in self.clips if c["id"] == cid)

    def next_id(self):
        return max((c["id"] for c in self.clips), default=0) + 1

    def set_range(self, c, start, end):
        """Move a clip to [start, end] seconds, snapping to the words inside."""
        dur = self.data["info"]["duration"]
        start, end = max(0.0, float(start)), min(dur, float(end))
        if end - start < 1:
            raise ValueError("clip must be at least 1 second long")
        c["i0"], c["i1"] = words_in(self.words, start, end)
        c["start"], c["end"] = round(start, 3), round(end, 3)

    def render(self, c, settings=None):
        s = settings or self.settings
        s = Settings.from_dict({**s.to_dict(), **c.get("overrides", {})})
        video = Path(self.data["video"])
        rank = sorted(self.clips, key=lambda x: -x.get("score", 0)).index(c) + 1
        name = f"{rank:02d}_{slug(c.get('title', 'clip'))}.mp4"
        old = c.get("file")
        for ext in (".mp4", ".jpg", ".txt", ".srt"):     # previous render of this clip
            if old and old != name:
                (self.dir / old).with_suffix(ext).unlink(missing_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        r = render_clip(video, self.data["info"], c, self.words, self.data.get("language", ""),
                        self.dir / name, self.work, s, c["id"])
        c.update(r, file=name, rendered=time.time())
        (self.dir / name).with_suffix(".txt").write_text(
            f"{c.get('title', '')}\n\n{c.get('caption', '')}\n\n" + " ".join("#" + h for h in c.get("hashtags", [])) + "\n")
        return c


def create_project(source, out_root, s: Settings, manual=None, reselect=False, retranscribe=False):
    """Download/probe/transcribe/pick. Returns a saved Project (not rendered yet)."""
    out_root = Path(out_root)
    if re.match(r"https?://", str(source)):
        source = fetch_url(str(source), out_root.parent / "downloads")
    video = Path(source).expanduser()
    if not video.exists():
        raise SystemExit(f"{video}: not found")
    if not FACE_MODEL.exists():
        raise SystemExit("Run ./setup.sh first (downloads the face model and caption font).")
    info = probe(video)
    p = Project(out_root / slug(video.stem))
    p.work.mkdir(parents=True, exist_ok=True)
    log(f"{video.name}: {info['w']}x{info['h']} @ {info['fps']:.2f} fps, {fmt_ts(info['duration'])} long", 0.01)

    tr = transcribe(video, p.work, s.whisper, s.language, retranscribe)
    words = tr["words"]
    tname = tr["file"]
    language = tr["language"]
    sents = sentences_from_words(words)
    (p.work / "transcript.txt").write_text("\n".join(f"[{i}] ({fmt_ts(x['s'])}) {x['text']}" for i, x in enumerate(sents)))
    if not words and not manual:
        raise SystemExit("No speech found in this video (use --clips to cut it by hand).")

    if manual:
        picks = []
        for k, (a, b) in enumerate(manual):
            b = min(b, info["duration"])
            i0, i1 = words_in(words, a, b)
            text = " ".join(words[i]["w"] for i in range(i0, i1 + 1)) if i0 is not None else ""
            picks.append({"i0": i0, "i1": i1, "start": a, "end": b, "score": 0, "scores": {},
                          "title": " ".join(text.split()[:7]) or f"Clip {k + 1}", "caption": text[:150],
                          "hashtags": [], "keywords": [], "why": "picked by hand"})
    else:
        sel_key = hashlib.sha1(json.dumps([s.model, s.count, s.min, s.max, s.think, s.prompt, len(words), 2]).encode()).hexdigest()[:10]
        sel_cache = p.work / f"picks-{sel_key}.json"
        if sel_cache.exists() and not reselect:
            picks = json.loads(sel_cache.read_text())
            log(f"[2/4] using the AI's previous picks ({sel_cache.name}; --reselect to ask again)", 0.4)
        else:
            chosen = pick_clips(sents, s, LANGS.get(language, language))
            picks = [{"i0": sents[c["a"]]["i0"], "i1": sents[c["b"]]["i1"],
                      **{k: c[k] for k in ("score", "scores", "title", "caption", "hashtags", "keywords", "why")}}
                     for c in chosen]
            sel_cache.write_text(json.dumps(picks, indent=1, ensure_ascii=False))
    if not picks:
        raise SystemExit("No good moments found. Try a lower --min, a higher --max, another --prompt, or --clips.")
    for k, c in enumerate(picks, 1):
        c["id"] = k
        if "start" not in c:
            c["start"], c["end"] = clip_bounds(words, c["i0"], c["i1"], info["duration"])
    old = {(c.get("i0"), c.get("i1")): c for c in p.clips}   # keep edits/renders of unchanged picks
    clips = []
    for c in picks:
        prev = old.get((c["i0"], c["i1"]))
        clips.append({**c, **{k: prev[k] for k in ("file", "overrides", "layout", "length", "cut_s") if k in prev}}
                     if prev else c)
    kept_files = {c.get("file") for c in clips}
    for c in p.clips:                                          # renders of picks that are gone
        if c.get("file") and c["file"] not in kept_files:
            for ext in (".mp4", ".jpg", ".txt", ".srt"):
                (p.dir / c["file"]).with_suffix(ext).unlink(missing_ok=True)
    p.data.update(video=str(video.resolve()), info=info, language=language, transcript=tname,
                  settings=s.to_dict(), clips=clips, created=p.data.get("created", time.time()))
    p.save()
    log(f"[2/4] {len(clips)} moments:", 0.45)
    for c in clips:
        log(f"   {c['id']}. [{c['score']:>3}] {fmt_ts(c['start'])}-{fmt_ts(c['end'])} ({c['end'] - c['start']:.0f}s)  {c['title']}")
    return p


def render_all(p: Project, only=None):
    clips = [c for c in p.clips if only is None or c["id"] in only]
    done = 0
    for c in sorted(clips, key=lambda c: -c.get("score", 0)):
        log(f"[3/4] rendering {done + 1}/{len(clips)}: {c['title']}", 0.45 + 0.55 * done / len(clips))
        try:
            p.render(c)
            log(f"       {c['length']:.0f}s, {c['layout']}, cut {c['cut_s']:.1f}s of pauses/fillers, "
                f"{c['render_s']:.0f}s to render")
        except RuntimeError as e:
            c["error"] = str(e)[-500:]
            log(str(e))
        done += 1
        p.save()
    write_gallery(p)
    return p


# ---------------------------------------------------------------- static gallery (the web app is richer)

def write_gallery(p: Project):
    cards = []
    for c in sorted([c for c in p.clips if c.get("file")], key=lambda c: -c.get("score", 0)):
        tags = " ".join("#" + h for h in c.get("hashtags", []))
        post = f"{c.get('caption', '')}\n\n{tags}"
        bars = "".join(f"<span>{k} {v}</span>" for k, v in c.get("scores", {}).items())
        cards.append(f"""<article>
  <video src="{html.escape(c['file'])}" poster="{html.escape(Path(c['file']).with_suffix('.jpg').name)}" controls preload="none" playsinline></video>
  <div class="meta">
    <div class="row"><span class="score s{c.get('score', 0) // 10 * 10}">{c.get('score', 0)}</span><h2>{html.escape(c.get('title', ''))}</h2></div>
    <p class="sub">{fmt_ts(c['start'])}–{fmt_ts(c['end'])} in source · {c.get('length', 0):.0f}s · {html.escape(c.get('layout', ''))}</p>
    <p class="bars">{bars}</p>
    <p class="why">{html.escape(c.get('why', ''))}</p>
    <pre>{html.escape(post)}</pre>
    <div class="row"><button data-copy="{html.escape(post)}">Copy caption</button><a href="{html.escape(c['file'])}" download>Download</a></div>
  </div>
</article>""")
    name = Path(p.data.get("video", "")).name
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Clips · {html.escape(name)}</title>
<style>
:root{{--bg:#0e0f12;--card:#17191e;--fg:#eceef2;--mut:#9aa0ab;--acc:#ffe500}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}}
header{{padding:24px 16px 8px;max-width:1200px;margin:auto}} h1{{margin:0;font-size:22px}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:18px;padding:16px;max-width:1200px;margin:auto}}
article{{background:var(--card);border-radius:14px;overflow:hidden}}
video{{width:100%;aspect-ratio:9/16;background:#000;display:block}}
.meta{{padding:12px 14px 16px}} .row{{display:flex;gap:10px;align-items:center}} h2{{font-size:16px;margin:0}}
.score{{font-weight:800;border-radius:8px;padding:2px 8px;background:#333}} .s90,.s100{{background:var(--acc);color:#000}} .s80{{background:#9be15d;color:#000}} .s70{{background:#5dc3e1;color:#000}}
.sub,.why,.bars{{color:var(--mut);margin:6px 0}} .bars span{{margin-right:10px}} pre{{white-space:pre-wrap;background:#0b0c0f;padding:10px;border-radius:8px;font:13px/1.4 system-ui}}
button,a{{background:#2a2d35;color:var(--fg);border:0;border-radius:8px;padding:7px 12px;font:inherit;cursor:pointer;text-decoration:none}}
</style></head><body><header><h1>{len(cards)} shorts from {html.escape(name)}</h1></header><main>
{chr(10).join(cards)}
</main><script>
document.querySelectorAll('button[data-copy]').forEach(b=>b.onclick=async()=>{{
 try{{await navigator.clipboard.writeText(b.dataset.copy);b.textContent='Copied!'}}catch(e){{b.textContent='Copy failed'}}
 setTimeout(()=>b.textContent='Copy caption',1500)}});
</script></body></html>"""
    (p.dir / "index.html").write_text(page, encoding="utf-8")


# ---------------------------------------------------------------- CLI

def main():
    d = Settings()
    p = argparse.ArgumentParser(description="Turn a long video (file or URL) into shorts for Instagram Reels.")
    p.add_argument("video", help="video file, or a YouTube/other URL")
    p.add_argument("-o", "--out", type=Path, default=Path("clips"), help="projects folder (default ./clips)")
    p.add_argument("-n", "--count", type=int, help=f"how many shorts to make (default {d.count})")
    p.add_argument("--min", type=float, help=f"shortest clip in seconds (default {d.min:g})")
    p.add_argument("--max", type=float, help=f"longest clip in seconds (default {d.max:g})")
    p.add_argument("--prompt", help="what to look for, e.g. 'moments about pricing'")
    p.add_argument("--clips", help="skip the AI and cut these ranges, e.g. '1:23-2:05,10:00-10:45'")
    p.add_argument("--model", help=f"Ollama model that picks the moments (default {d.model})")
    p.add_argument("--think", action="store_true", default=None, help="let the model reason first (slower)")
    p.add_argument("--whisper", help=f"faster-whisper model (default {d.whisper})")
    p.add_argument("--language", help="spoken language code, e.g. en (default: detect)")
    p.add_argument("--aspect", choices=list(ASPECTS), help="9:16 Reels (default), 4:5 feed, 1:1 square")
    p.add_argument("--layout", choices=["auto", "track", "split", "fit", "center"])
    p.add_argument("--captions", choices=list(CAPTION_STYLES) + ["none"])
    p.add_argument("--caption-pos", choices=["auto", "low", "middle"])
    p.add_argument("--highlight", help="active-word color, e.g. '#FFE500'")
    p.add_argument("--emphasis", help="keyword color, e.g. '#39FF14'")
    p.add_argument("--no-hook", dest="hook", action="store_false", default=None, help="no title card at the start")
    p.add_argument("--no-progress-bar", dest="progress_bar", action="store_false", default=None)
    p.add_argument("--no-zoom", dest="zoom", action="store_false", default=None, help="no punch-in zooms")
    p.add_argument("--keep-fillers", dest="fillers", action="store_false", default=None, help="don't cut um/uh")
    p.add_argument("--max-gap", type=float, help="cut pauses longer than this many seconds (0 = keep all)")
    p.add_argument("--logo", help="PNG logo to overlay")
    p.add_argument("--logo-pos", choices=["top-left", "top-right", "bottom-left", "bottom-right"])
    p.add_argument("--music", help="background music file or folder (ducked under the voice)")
    p.add_argument("--music-volume", type=float)
    p.add_argument("--font-file", help="caption font (.ttf/.otf)")
    p.add_argument("--brand", default=str(HERE / "brand.json"), help="JSON of default settings (fonts, colors, logo...)")
    p.add_argument("--encoder", choices=["auto", "nvenc", "x264"])
    p.add_argument("--dry-run", action="store_true", help="pick the moments and stop before rendering")
    p.add_argument("--reselect", action="store_true", help="ask the AI again instead of reusing its last picks")
    p.add_argument("--retranscribe", action="store_true")
    args = p.parse_args()

    base = {**Settings().to_dict(), **load_brand(args.brand)}
    over = {k: v for k, v in vars(args).items() if v is not None and k in base}
    s = Settings.from_dict({**base, **over})
    manual = None
    if args.clips:
        manual = [tuple(parse_ts(x) for x in rng.split("-")) for rng in args.clips.split(",")]
    t0 = time.time()
    proj = create_project(args.video, args.out, s, manual, args.reselect, args.retranscribe)
    if args.dry_run:
        return
    render_all(proj)
    ok = sum(1 for c in proj.clips if c.get("file"))
    log(f"[4/4] done in {time.time() - t0:.0f}s: {ok} shorts in {proj.dir}/ (open {proj.dir / 'index.html'})", 1.0)


if __name__ == "__main__":
    main()
