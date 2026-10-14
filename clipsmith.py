#!/usr/bin/env -S uv run --project /home/jt/claude/clipsmith python
"""Designer Studio (engine + CLI): our own Opus Clip. Turns a long video (file or URL) into ready-to-post shorts.

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
_checks = []


def checkpoint():
    """Where a job can be paused or stopped (the web app hooks in; a no-op on the command line)."""
    for c in _checks:
        c()


def log(msg, progress=None):
    checkpoint()
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
    scenes: bool = True              # reset framing at camera cuts, layout per scene
    vision: bool = True              # a vision model looks at each scene (layout, burned-in subtitles)
    vision_model: str = "qwen3.5:9b"

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
            words = []
            for seg in segments:
                checkpoint()
                words += [{"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)}
                          for w in (seg.words or []) if w.word.strip()]
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
Instagram Reels, TikTok and YouTube Shorts. You get a timestamped transcript, one numbered line per \
sentence. Pick the best self-contained moments to cut into vertical shorts.

What makes a great short:
- HOOK: the first line grabs attention within 2 seconds (a bold claim, a question, a surprising fact, \
conflict, a strong emotion, a punchline setup). It must make sense with no earlier context: never start \
on "And", "So", "But", "That's why", or on a pronoun that refers to something said before.
- PAYOFF: it ends on a complete thought: the answer, punchline, lesson or twist. Never stop mid-story.
- One clear idea per clip. Story, insight, hot take, funny exchange, emotional peak, or useful tip.
- Prefer moments people would rewatch, share, save or comment on.
{context}{focus}
Rules:
- Each clip spans whole lines: start_line to end_line (inclusive), using the line numbers shown.
- Each clip lasts between {min_s} and {max_s} seconds (use the timestamps). Clips must not overlap.
- Return up to {n} clips, best first. Fewer is fine if the material is weak; never pad with filler.
- Scores are honest 0-100 estimates: hook (first 2 seconds), flow (coherent, complete), value \
(insight, emotion or entertainment), trend (fits what performs on short-form now), and score = overall \
viral potential (90+ only for exceptional moments).
- hooks: 3 different on-screen hook texts for the first seconds (max 8 words each, punchy, no hashtags, \
no emojis), each a different angle: curiosity gap, bold claim, direct question or "you" benefit. Best first.
- instagram: caption (1-2 short sentences that tease the clip plus a light call to action, e.g. save or \
comment) and 5-8 hashtags.
- tiktok: caption (one short, casual line, may end with a question to drive comments) and 3-5 hashtags.
- youtube: title (max 70 characters, searchable, curiosity-driven), description (1-2 sentences) and \
3-6 tags.
- Hashtags and tags without the # sign.
- keywords: 3-6 single words, copied exactly as spoken in the clip, that carry the punch (numbers, \
strong nouns, emotional words). They get highlighted in the captions.
- topic: 1-3 words naming the clip's theme (for tracking which topics perform).
- why: one sentence on why this moment works.
- Write hooks, captions, titles, descriptions, hashtags and topic in the same language as the \
transcript ({language}), in the speaker's own voice and vocabulary. Write "why" in English."""

_STR = {"type": "string"}
_TAGS = {"type": "array", "items": _STR}
CLIP_SCHEMA = {
    "type": "object",
    "properties": {"clips": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "start_line": {"type": "integer"}, "end_line": {"type": "integer"},
            "hook": {"type": "integer"}, "flow": {"type": "integer"}, "value": {"type": "integer"},
            "trend": {"type": "integer"}, "score": {"type": "integer"}, "hooks": _TAGS,
            "instagram": {"type": "object", "properties": {"caption": _STR, "hashtags": _TAGS}, "required": ["caption", "hashtags"]},
            "tiktok": {"type": "object", "properties": {"caption": _STR, "hashtags": _TAGS}, "required": ["caption", "hashtags"]},
            "youtube": {"type": "object", "properties": {"title": _STR, "description": _STR, "tags": _TAGS},
                        "required": ["title", "description", "tags"]},
            "keywords": _TAGS, "topic": _STR, "why": _STR},
        "required": ["start_line", "end_line", "hook", "flow", "value", "trend", "score", "hooks", "instagram",
                     "tiktok", "youtube", "keywords", "topic", "why"]}}},
    "required": ["clips"],
}

MAP_PROMPT = """You analyze the structure of a long video from its timestamped transcript (one numbered \
line per sentence) for a short-form editor. Split it into consecutive chapters covering the whole \
transcript in order (roughly 30 seconds to 3 minutes each, following the topic changes). For each chapter:
- title: 2-6 words; summary: one sentence (both in {language}).
- engagement: honest 0-100 estimate of how gripping it is for short-form viewers (hooks, emotion, \
surprise, stakes, humor, useful insight; low for setup, rambling, logistics).
- emotion: one word in English (e.g. inspiring, funny, tense, vulnerable, informative, surprising, flat).
- peak_line: the single most clip-worthy line in the chapter, and peak_reason: a short English phrase why."""

MAP_SCHEMA = {
    "type": "object",
    "properties": {"chapters": {"type": "array", "items": {
        "type": "object",
        "properties": {"start_line": {"type": "integer"}, "end_line": {"type": "integer"}, "title": _STR,
                       "summary": _STR, "engagement": {"type": "integer"}, "emotion": _STR,
                       "peak_line": {"type": "integer"}, "peak_reason": _STR},
        "required": ["start_line", "end_line", "title", "summary", "engagement", "emotion", "peak_line",
                     "peak_reason"]}}},
    "required": ["chapters"],
}


def _post(url, body, timeout):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ensure_model_on_gpu(url, model, num_ctx):
    """Load the model and make sure Ollama put all of it in VRAM (not partly on the CPU)."""
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


def ollama_chat(url, model, system, user, think, num_ctx, schema=CLIP_SCHEMA, temperature=0.4):
    body = {"model": model, "stream": False, "format": schema, "think": think, "keep_alive": "5m",
            "options": {"temperature": temperature, "num_ctx": num_ctx},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if schema is None:
        body.pop("format")
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


def transcript_windows(sents, window_s=15 * 60, overlap_s=90):
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
    return windows


def context_size(sents, windows, answer_tokens=6000):
    """Context sized to the longest window (about 3 chars per token, plus prompt and answer), so the model
    and its cache fit in VRAM."""
    longest = max(sum(len(sents[i]["text"]) + 12 for i in range(a, b + 1)) for a, b in windows)
    return min(32768, max(8192, math.ceil((longest / 3 + answer_tokens) / 2048) * 2048))


def numbered(sents, a, b):
    return "\n".join(f"[{i}] ({fmt_ts(sents[i]['s'])}) {sents[i]['text']}" for i in range(a, b + 1))


def video_map(sents, s: Settings, language):
    """Engagement map of the whole video: chapters with a summary, engagement score, emotion and peak."""
    windows = transcript_windows(sents, window_s=20 * 60, overlap_s=0)
    url = s.ollama.rstrip("/")
    num_ctx = context_size(sents, windows, 4000)
    ensure_model_on_gpu(url, s.model, num_ctx)
    chapters = []
    for wi, (a, b) in enumerate(windows):
        log(f"[2/4] mapping the video's structure ({wi + 1}/{len(windows)})...", 0.26)
        raw = ollama_chat(url, s.model, MAP_PROMPT.format(language=language), f"Transcript:\n{numbered(sents, a, b)}",
                          False, num_ctx, MAP_SCHEMA, 0.2)
        try:
            got = json.loads(raw).get("chapters", [])
        except json.JSONDecodeError:
            got = []
        for c in got:
            lo, hi = sorted((int(c["start_line"]), int(c["end_line"])))
            lo, hi = max(a, lo), min(b, hi)
            if lo > hi:
                continue
            pk = int(c["peak_line"]) if lo <= int(c["peak_line"]) <= hi else lo
            chapters.append({"a": lo, "b": hi, "start": sents[lo]["s"], "end": sents[hi]["e"],
                             "title": c["title"].strip(), "summary": c["summary"].strip(),
                             "engagement": max(0, min(100, int(c["engagement"]))), "emotion": c["emotion"].strip().lower(),
                             "peak": {"line": pk, "start": sents[pk]["s"], "end": sents[pk]["e"],
                                      "text": sents[pk]["text"], "reason": c["peak_reason"].strip()}})
    chapters.sort(key=lambda c: c["a"])
    log(f"[2/4] video map: {len(chapters)} chapters", 0.3)
    return chapters


def pick_clips(sents, s: Settings, language, context=""):
    """Ask the LLM for candidate moments, window by window for long videos; then rank and de-overlap."""
    windows = transcript_windows(sents)
    url = s.ollama.rstrip("/")
    num_ctx = context_size(sents, windows) + (2048 if context else 0)
    ensure_model_on_gpu(url, s.model, min(32768, num_ctx))
    per_window = max(3, math.ceil(s.count * 1.5 / len(windows)) + 1)
    focus = (f"\nThe creator wants clips about: {s.prompt}\nOnly pick moments that match this; return fewer "
             "clips rather than off-topic ones.\n") if s.prompt else ""
    system = EDITOR_PROMPT.format(min_s=s.min, max_s=s.max, n=per_window, language=language, focus=focus,
                                  context=context)
    cands = []
    clamp = lambda v: max(0, min(100, int(v)))
    tags = lambda xs: [h.strip().lstrip("#").replace(" ", "") for h in xs if h.strip()]
    for wi, (a, b) in enumerate(windows):
        t0 = time.time()
        log(f"[2/4] {s.model} reading transcript part {wi + 1}/{len(windows)} "
            f"({fmt_ts(sents[a]['s'])}-{fmt_ts(sents[b]['e'])})...", 0.3 + 0.12 * wi / len(windows))
        raw = ollama_chat(url, s.model, system, f"Transcript:\n{numbered(sents, a, b)}", s.think, min(32768, num_ctx))
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
            hooks = [h.strip().strip('"') for h in c["hooks"] if h.strip()][:3] or ["(no hook)"]
            ig, tt, yt = c["instagram"], c["tiktok"], c["youtube"]
            platforms = {"instagram": {"caption": ig["caption"].strip(), "hashtags": tags(ig["hashtags"])},
                         "tiktok": {"caption": tt["caption"].strip(), "hashtags": tags(tt["hashtags"])},
                         "youtube": {"title": yt["title"].strip(), "description": yt["description"].strip(),
                                     "tags": tags(yt["tags"])}}
            c = fit_length({"a": lo, "b": hi, "score": clamp(c["score"]),
                            "scores": {k: clamp(c[k]) for k in ("hook", "flow", "value", "trend")},
                            "title": hooks[0], "hooks": hooks, "platforms": platforms,
                            "caption": platforms["instagram"]["caption"], "hashtags": platforms["instagram"]["hashtags"],
                            "keywords": [k.strip() for k in c["keywords"] if k.strip()][:6],
                            "topic": c["topic"].strip(), "why": c["why"].strip()}, sents, s.min, s.max)
            if c:
                cands.append(c)

    chosen = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        a, b = sents[c["a"]]["s"], sents[c["b"]]["e"]
        if all(min(b, sents[o["b"]]["e"]) - max(a, sents[o["a"]]["s"]) < 0.25 * (b - a) for o in chosen):
            chosen.append(c)
        if len(chosen) >= s.count:
            break
    return chosen


# ---------------------------------------------------------------- remember & learn: creator profile + past results

PROFILE_FILE = HERE / "profile.json"
PROFILE_FIELDS = {"niche": "", "audience": "", "tone": "", "themes": "", "avoid": "", "goals": ""}


def load_profile():
    try:
        return {**PROFILE_FIELDS, **json.loads(PROFILE_FILE.read_text())}
    except (OSError, json.JSONDecodeError):
        return dict(PROFILE_FIELDS)


def save_profile(d):
    PROFILE_FILE.write_text(json.dumps({k: str(d.get(k, "")).strip() for k in PROFILE_FIELDS}, indent=1, ensure_ascii=False))


def engagement_rate(m):
    v = float(m.get("views") or 0)
    inter = sum(float(m.get(k) or 0) for k in ("likes", "comments", "shares", "saves"))
    return inter / v if v > 0 else 0.0


def posted_clips(root):
    """Every clip (in every project under root) that has performance numbers entered."""
    out = []
    for f in Path(root).glob("*/project.json"):
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for c in d.get("clips", []):
            for m in c.get("metrics", []):
                if m.get("views"):
                    out.append({"project": f.parent.name, "video": Path(d.get("video", "")).name, "clip": c, "m": m,
                                "rate": engagement_rate(m), "length": c.get("length") or (c["end"] - c["start"]),
                                "style": {**d.get("settings", {}), **c.get("overrides", {})}})
    return out


def learning_context(root):
    """Creator profile + what has worked before, for the picker prompt."""
    prof = load_profile()
    lines = []
    if any(prof.values()):
        lines.append("Creator profile (pick and write for this):")
        for k, label in (("niche", "Niche"), ("audience", "Audience"), ("tone", "Tone"), ("themes", "Themes"),
                         ("goals", "Goals"), ("avoid", "Avoid")):
            if prof[k]:
                lines.append(f"- {label}: {prof[k]}")
    posted = posted_clips(root)
    if len(posted) >= 3:
        posted.sort(key=lambda x: -float(x["m"]["views"]))
        fmt = lambda x: (f"- \"{x['clip'].get('title', '')}\" (topic: {x['clip'].get('topic', '?')}, "
                         f"{x['length']:.0f}s, {x['m'].get('platform', '?')}): {int(float(x['m']['views'])):,} views, "
                         f"{x['rate'] * 100:.1f}% engagement")
        best, worst = posted[:5], posted[max(5, len(posted) - 3):]   # never the same clip in both
        lines.append("Past clips that performed best (learn from their hooks and topics):")
        lines += [fmt(x) for x in best]
        if worst:
            lines.append("Past clips that performed worst (avoid what they did):")
            lines += [fmt(x) for x in worst]
    return ("\n" + "\n".join(lines) + "\n") if lines else ""


INSIGHT_PROMPT = """You are a short-form growth analyst. Below is a creator's posted clips with their \
stats, followed by breakdowns by topic, length, caption style, platform and posting hour. In plain, \
specific English, give: (1) what is working, (2) what is not, (3) five concrete recommendations for the \
next videos (hooks, topics, length, format, timing). Keep it under 250 words. Use short bullet points. \
Base everything on the numbers; say when the data is too thin to tell."""


def insights(root, s: Settings = None):
    """Breakdowns of performance (and optionally an AI summary)."""
    posted = posted_clips(root)

    def group(keyf):
        g = {}
        for x in posted:
            k = keyf(x)
            if k is None:
                continue
            g.setdefault(k, []).append(x)
        rows = [{"key": k, "clips": len(v), "median_views": float(np.median([float(x["m"]["views"]) for x in v])),
                 "avg_rate": float(np.mean([x["rate"] for x in v]))} for k, v in g.items()]
        return sorted(rows, key=lambda r: -r["median_views"])

    def hour(x):
        at = x["m"].get("posted_at") or ""
        m = re.search(r"T?(\d{1,2}):\d{2}", at)
        return f"{int(m.group(1)):02d}:00" if m else None

    bucket = lambda x: "<20s" if x["length"] < 20 else "20-35s" if x["length"] < 35 else "35-50s" if x["length"] < 50 else "50s+"
    data = {"count": len(posted),
            "total_views": sum(float(x["m"]["views"]) for x in posted),
            "by_topic": group(lambda x: (x["clip"].get("topic") or "?").lower()),
            "by_length": group(bucket),
            "by_captions": group(lambda x: x["style"].get("captions", "bold")),
            "by_platform": group(lambda x: x["m"].get("platform") or "?"),
            "by_hour": group(hour),
            "top": [{"title": x["clip"].get("title"), "project": x["project"], "platform": x["m"].get("platform"),
                     "views": float(x["m"]["views"]), "rate": x["rate"], "topic": x["clip"].get("topic"),
                     "length": x["length"]} for x in sorted(posted, key=lambda x: -float(x["m"]["views"]))[:20]]}
    if s is not None:
        if len(posted) < 3:
            data["ai"] = "Enter stats for at least 3 posted clips to get AI insights."
        else:
            table = "\n".join(f"- {t['title']} | topic {t['topic']} | {t['length']:.0f}s | {t['platform']} | "
                              f"{int(t['views']):,} views | {t['rate'] * 100:.1f}% engagement" for t in data["top"])
            br = "\n".join(f"{name}: " + "; ".join(f"{r['key']} = {r['clips']} clips, median {int(r['median_views']):,} views, "
                                                  f"{r['avg_rate'] * 100:.1f}% eng" for r in data[k])
                           for name, k in (("Topic", "by_topic"), ("Length", "by_length"), ("Captions", "by_captions"),
                                           ("Platform", "by_platform"), ("Posting hour", "by_hour")))
            url = s.ollama.rstrip("/")
            ensure_model_on_gpu(url, s.model, 8192)
            data["ai"] = ollama_chat(url, s.model, INSIGHT_PROMPT, f"Clips:\n{table}\n\nBreakdowns:\n{br}", False, 8192,
                                     None, 0.3).strip()
            ollama_unload(url, s.model)
    return data


# ---------------------------------------------------------------- clip sheet + posting schedule

DEFAULT_SLOTS = {"instagram": ["11:00", "19:00"], "tiktok": ["12:00", "20:00"], "youtube": ["15:00"]}


def best_slots(root, platform):
    """Posting times: learned from your own stats when there is enough data, else common good defaults."""
    plat = [x for x in posted_clips(root) if (x["m"].get("platform") or "") == platform]
    if len(plat) >= 6:
        hours = {}
        for x in plat:
            m = re.search(r"T?(\d{1,2}):\d{2}", x["m"].get("posted_at") or "")
            if m:
                hours.setdefault(int(m.group(1)), []).append(float(x["m"]["views"]))
        good = sorted(((float(np.median(v)), h) for h, v in hours.items() if len(v) >= 2), reverse=True)
        if len(good) >= 2:
            return [f"{h:02d}:00" for _, h in good[:2]], "learned from your stats"
    return DEFAULT_SLOTS.get(platform, ["12:00"]), "typical best times"


def schedule(p, start_date=None, per_day=1, platforms=("instagram",), root=None):
    """Spread the clips (best first) over the coming days, one slot per platform per post."""
    import datetime as dt
    d0 = dt.date.fromisoformat(start_date) if start_date else dt.date.today() + dt.timedelta(days=1)
    clips = sorted(p.clips, key=lambda c: -c.get("score", 0))
    slots = {pl: best_slots(root or p.dir.parent, pl) for pl in platforms}
    plan = []
    for k, c in enumerate(clips):
        day = d0 + dt.timedelta(days=k // max(1, per_day))
        for pl in platforms:
            times, why = slots[pl]
            plan.append({"clip": c["id"], "platform": pl, "date": day.isoformat(),
                         "time": times[(k % per_day) % len(times)], "why": why})
    return plan


def clip_sheet(p, plan=None):
    """Rows for the clip sheet (one per clip and platform when a schedule is given)."""
    rows = []
    sched = {}
    for e in plan or []:
        sched.setdefault(e["clip"], []).append(e)
    for rank, c in enumerate(sorted(p.clips, key=lambda c: -c.get("score", 0)), 1):
        pf = c.get("platforms") or {"instagram": {"caption": c.get("caption", ""), "hashtags": c.get("hashtags", [])}}
        for e in sched.get(c["id"], [{"platform": None}]):
            pl = e["platform"]
            info = pf.get(pl or "instagram", {})
            caption = info.get("caption") or info.get("description") or c.get("caption", "")
            tags = info.get("hashtags") or info.get("tags") or c.get("hashtags", [])
            rows.append({"rank": rank, "file": c.get("file", ""), "source_start": fmt_ts(c["start"]),
                         "source_end": fmt_ts(c["end"]), "length_s": round(c.get("length") or c["end"] - c["start"]),
                         "score": c.get("score", 0), **{k: c.get("scores", {}).get(k, "") for k in ("hook", "flow", "value", "trend")},
                         "topic": c.get("topic", ""), "hook_text": c.get("title", ""),
                         "alt_hooks": " | ".join(h for h in c.get("hooks", []) if h != c.get("title")),
                         "platform": pl or "", "post_date": e.get("date", ""), "post_time": e.get("time", ""),
                         "title": info.get("title", "") if pl == "youtube" else "",
                         "caption": caption, "hashtags": " ".join("#" + h for h in tags), "why": c.get("why", "")})
    return rows


def write_sheet_csv(path, rows):
    import csv
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # BOM so Excel reads accents right
        if not rows:
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


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
        checkpoint()
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


# ---------------------------------------------------------------- scenes: cuts, what each shot shows, its layout

def detect_scenes(video, cs, dur, min_len=1.2):
    """Camera cuts inside [cs, cs+dur] as clip-relative seconds (ffmpeg scdet, GPU decode)."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-hwaccel", "cuda", "-ss", f"{cs:.3f}", "-t", f"{dur:.3f}",
                        "-i", str(video), "-vf", "scale=320:-2,scdet=threshold=14", "-an", "-f", "null", "-"],
                       capture_output=True, text=True)
    cuts = []
    for m in re.finditer(r"lavfi\.scd\.time:\s*([\d.]+)", r.stderr):
        t = float(m.group(1))
        if t - (cuts[-1] if cuts else 0.0) >= min_len and dur - t >= min_len:
            cuts.append(t)
    return cuts


VISION_PROMPT = """Look at this frame from a video that will be reframed to vertical 9:16. Answer in JSON.
kind: one_person (one main person on camera), two_people (two people side by side or in one shot), group \
(three or more), screen_share (a screen, slides, app or game with no visible person), screen_with_webcam \
(screen content plus a small camera view of a person), or broll_or_other (anything else).
burned_subtitles: true only if the frame already shows subtitle/caption text burned into the picture."""
VISION_SCHEMA = {"type": "object", "properties": {
    "kind": {"type": "string", "enum": ["one_person", "two_people", "group", "screen_share", "screen_with_webcam",
                                        "broll_or_other"]},
    "burned_subtitles": {"type": "boolean"}}, "required": ["kind", "burned_subtitles"]}


def vision_scenes(video, cs, scenes, s: Settings, work):
    """Ask the vision model what each scene shows. Cached per clip range; [] when vision is unavailable."""
    import base64
    key = hashlib.sha1(json.dumps([str(video), round(cs, 2), [[round(a, 2), round(b, 2)] for a, b in scenes],
                                   s.vision_model]).encode()).hexdigest()[:12]
    cache = work / f"vision-{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    url = s.ollama.rstrip("/")
    ensure_model_on_gpu(url, s.vision_model, 4096)
    out = []
    for a, b in scenes:
        jpg = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{cs + (a + b) / 2:.3f}", "-i", str(video), "-frames:v", "1",
                              "-vf", "scale=768:-2", "-f", "image2", "-c:v", "mjpeg", "-q:v", "4", "-"],
                             capture_output=True).stdout
        try:
            r = _post(url + "/api/chat", {"model": s.vision_model, "stream": False, "format": VISION_SCHEMA, "think": False,
                                          "keep_alive": "5m", "options": {"temperature": 0, "num_ctx": 4096},
                                          "messages": [{"role": "user", "content": VISION_PROMPT,
                                                        "images": [base64.b64encode(jpg).decode()]}]}, 120)
            out.append(json.loads(r["message"]["content"]))
        except Exception as e:  # noqa: BLE001
            log(f"       vision check failed ({str(e)[:80]}); using face tracking only")
            return []
    cache.write_text(json.dumps(out))
    return out


def scene_stats(per_sample):
    stats = {}
    for faces in per_sample:
        for tid, cx, cy, h, _ in faces:
            st = stats.setdefault(tid, {"n": 0, "cx": [], "cy": [], "h": []})
            st["n"] += 1
            st["cx"].append(cx)
            st["cy"].append(cy)
            st["h"].append(h)
    return stats


def render_clip(video, info, clip, words, language, out_path, work, s: Settings, k=0, explicit_captions=False, dub=None):
    """dub: {"wav", "words", "title"} renders the clip with a new voice track, captions and hook (dub.py)."""
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
    title = clip.get("title", "")
    if dub:
        ws, title = dub["words"], dub["title"]

    # base window: the largest output-shaped rectangle that fits in the source
    if W / H > a_out:
        bw, bh = H * a_out, float(H)
    else:
        bw, bh = float(W), W / a_out
    wide = W / H > a_out * 1.2
    cap_y = int(oh * (0.71 if s.aspect == "9:16" else 0.8))
    if s.caption_pos == "middle":
        cap_y = int(oh * 0.58)
    hook_y = int(oh * 0.13)
    inputs = ["-hwaccel", "cuda", "-ss", f"{cs:.3f}", "-t", f"{dur:.3f}", "-i", str(Path(video).resolve())]

    # ---- scenes and what each one shows
    cuts = detect_scenes(video, cs, dur) if s.scenes and s.layout != "fit" else []
    bounds = [0.0] + cuts + [dur]
    scenes = list(zip(bounds, bounds[1:]))
    seen = vision_scenes(video, cs, scenes, s, work) if (s.vision and wide and s.layout in ("auto", "screen")) else []
    burned = bool(seen) and sum(1 for v in seen if v.get("burned_subtitles")) > len(seen) / 2

    samples = per_sample = None
    aw, ah = 640, int(round(640 * H / W / 2)) * 2
    if s.layout != "fit":
        samples, aw, ah = analyze_faces(video, info, cs, dur)
        per_sample, _ = track_faces(samples)
    times = np.array([t for t, _ in samples]) if samples else np.array([0.0])
    sc = W / aw                     # analysis px -> source px
    plan = []                       # per scene: {"layout", ...}
    for si, (a, b) in enumerate(scenes):
        idx = [i for i, t in enumerate(times) if a <= t < b] if per_sample else []
        sub = [per_sample[i] for i in idx] if per_sample else []
        stats = scene_stats(sub)
        faced = sum(1 for f in sub if f) / max(1, len(sub))
        kind = seen[si]["kind"] if si < len(seen) else None
        lay = s.layout
        if lay in ("auto", "split", "screen"):
            two = detect_split(sub, stats, aw, bw / sc) if (wide and sub) else None
            if lay == "screen" or kind in ("screen_share", "screen_with_webcam"):
                lay = "screen"
            elif two:                      # two faces far apart all scene long: trust the geometry
                lay = "split"
            elif lay == "split":
                lay = "track"
            else:
                lay = "track" if faced >= 0.35 else "fit"
        if lay == "screen":
            big = max(stats.items(), key=lambda kv: kv[1]["n"] * np.median(kv[1]["h"]), default=None)
            if big is None or big[1]["n"] < 0.3 * max(1, len(sub)):
                lay = "fit"            # nothing to put in the presenter strip
        e = {"a": a, "b": b, "layout": lay, "idx": idx, "kind": kind}
        if lay == "track":
            e["cam"] = plan_camera(sub, aw, ah, bw / sc) if faced > 0 else None
        elif lay == "split":
            e["split"] = [(float(np.median(stats[t]["cx"])), float(np.median(stats[t]["cy"])), float(np.median(stats[t]["h"])))
                          for t in two]
        elif lay == "screen":
            st = big[1]
            e["face"] = (float(np.median(st["cx"])), float(np.median(st["cy"])), float(np.median(st["h"])))
        plan.append(e)
    # neighbours with the same layout and no cut between them in practice: keep as separate scenes (camera resets)
    used = []
    for e in plan:
        if e["layout"] not in used:
            used.append(e["layout"])

    zf = zoom_curve(tl, kw_times, s.zoom and ("track" in used or "center" in used))
    scene_of = lambda t: next((i for i, e in enumerate(plan) if t < e["b"]), len(plan) - 1)
    frames = int(dur * fps) + 2
    branch = {}
    g0 = []

    def sendcmd_file(name, rows):
        f = work / f"{tag}.{name}.cmd"
        f.write_text("\n".join(rows) + "\n")
        return fesc(f.name)

    if "track" in used or "center" in used:
        lines, last, first = [], None, None
        for f in range(frames):
            t = f / fps
            e = plan[scene_of(t)]
            if e["layout"] in ("track", "center"):
                cam = e.get("cam")
                if cam is not None and e["idx"]:
                    tt = times[e["idx"]]
                    seg_cuts = [tt[i] for i in cam["cuts"]]
                    seg = sum(1 for c in seg_cuts if t >= c)
                    lo = 0 if seg == 0 else cam["cuts"][seg - 1]
                    hi = cam["cuts"][seg] if seg < len(cam["cuts"]) else len(tt)
                    fx = np.interp(t, tt[lo:hi], cam["x"][lo:hi]) * sc
                    fy = np.interp(t, tt[lo:hi], cam["y"][lo:hi]) * sc
                else:
                    fx, fy = W / 2, H * 0.4
                if e["layout"] == "center" or s.layout == "center":
                    fx = W / 2
                z = zf(t)
                w, h = bw / z, bh / z
                x = min(max(fx - w / 2, 0), W - w)
                y = min(max(fy - 0.38 * h, 0), H - h) if (z > 1.001 or bh < H) else (H - h) / 2
                v = (int(w) // 2 * 2, int(h) // 2 * 2, int(x) // 2 * 2, int(y) // 2 * 2)
            else:
                v = last or (int(bw) // 2 * 2, int(bh) // 2 * 2, 0, 0)
            if first is None:
                first = v
            if v != last:
                cmds = [f"crop@rf {nm} {val}" for nm, val, old in zip("whxy", v, last or (None,) * 4) if val != old]
                lines.append(f"{t:.4f} " + ", ".join(cmds) + ";")
                last = v
        branch["track"] = (f"sendcmd=f={sendcmd_file('track', lines)},crop@rf=w={first[0]}:h={first[1]}:x={first[2]}:y={first[3]},"
                           f"scale={ow}:{oh}:flags=lanczos")
        branch["center"] = branch["track"]

    if "split" in used:
        halves = {0: [], 1: []}
        firsts = {}
        for e in plan:
            if e["layout"] != "split":
                continue
            for j, (fx, fy, fh) in enumerate(e["split"]):
                fx, fy, fh = fx * sc, fy * sc, fh * sc
                ch = min(float(H), 3.4 * fh)
                cw = ch * ow / (oh / 2)
                if cw > W:
                    cw, ch = float(W), W * (oh / 2) / ow
                x = min(max(fx - cw / 2, 0), W - cw)
                y = min(max(fy - 0.42 * ch, 0), H - ch)
                v = (int(cw) // 2 * 2, int(ch) // 2 * 2, int(x) // 2 * 2, int(y) // 2 * 2)
                firsts.setdefault(j, v)
                halves[j].append(f"{max(0.0, e['a']):.4f} " + ", ".join(f"crop@h{j} {nm} {val}" for nm, val in zip("whxy", v)) + ";")
        parts = []
        for j in (0, 1):
            w0, h0, x0, y0 = firsts[j]
            parts.append(f"sendcmd=f={sendcmd_file(f'half{j}', halves[j])},crop@h{j}=w={w0}:h={h0}:x={x0}:y={y0},"
                         f"scale={ow}:{oh // 2}:flags=lanczos,setsar=1")
        branch["split"] = ("split[sa][sb];[sa]" + parts[0] + "[stop];[sb]" + parts[1] + "[sbot];[stop][sbot]vstack")
        cap_y = oh // 2 + 60 if s.caption_pos != "low" else cap_y    # captions on the seam, covering nobody
        hook_y = oh // 2 - 330

    if "screen" in used:
        h1 = int(oh * 0.56) // 2 * 2
        h2 = oh - h1
        rows, first = [], None
        for e in plan:
            if e["layout"] != "screen":
                continue
            fx, fy, fh = (v * sc for v in e["face"])
            ch = min(float(H), 2.6 * fh)
            cw = ch * ow / h2
            if cw > W:
                cw, ch = float(W), W * h2 / ow
            x = min(max(fx - cw / 2, 0), W - cw)
            y = min(max(fy - 0.45 * ch, 0), H - ch)
            v = (int(cw) // 2 * 2, int(ch) // 2 * 2, int(x) // 2 * 2, int(y) // 2 * 2)
            first = first or v
            rows.append(f"{e['a']:.4f} " + ", ".join(f"crop@pf {nm} {val}" for nm, val in zip("whxy", v)) + ";")
        branch["screen"] = (f"split=3[s1][s2][s3];[s1]scale={ow}:{h1}:force_original_aspect_ratio=increase,crop={ow}:{h1},"
                            f"gblur=sigma=30,eq=brightness=-0.15[sbg];[s2]scale={ow}:{h1}:force_original_aspect_ratio=decrease:"
                            f"flags=lanczos[sfg];[sbg][sfg]overlay=(W-w)/2:(H-h)/2,setsar=1[stop2];"
                            f"[s3]sendcmd=f={sendcmd_file('presenter', rows)},crop@pf=w={first[0]}:h={first[1]}:x={first[2]}:"
                            f"y={first[3]},scale={ow}:{h2}:flags=lanczos,setsar=1[sbot2];[stop2][sbot2]vstack")
        if "split" not in used and s.caption_pos != "low":
            cap_y = h1 + 120                                       # just under the screen

    if "fit" in used:
        branch["fit"] = (f"split[fb][ff];[fb]scale={ow}:{oh}:force_original_aspect_ratio=increase,crop={ow}:{oh},"
                         f"gblur=sigma=40,eq=brightness=-0.12[fbg];[ff]scale={ow}:{oh}:force_original_aspect_ratio=decrease:"
                         f"flags=lanczos[ffg];[fbg][ffg]overlay=(W-w)/2:(H-h)/2")

    # one branch per layout over the whole clip, then each scene takes its stretch from its layout's branch
    names = {lay: f"L{i}" for i, lay in enumerate(used)}
    if len(used) > 1:
        g0.append("[0:v]split=" + str(len(used)) + "".join(f"[in{names[l]}]" for l in used))
    else:
        g0.append(f"[0:v]null[in{names[used[0]]}]")
    for lay in used:
        chain = branch[lay]
        # a branch's internal labels must be unique: prefix them with the branch name
        chain = re.sub(r"\[(s[a-z0-9]+|f[a-z0-9]+|stop2?|sbot2?)\]", lambda m, n=names[lay]: f"[{n}{m.group(1)}]", chain)
        g0.append(f"[in{names[lay]}]{chain},setsar=1,fps={fps:.4f},format=yuv420p[{names[lay]}]")
    if len(plan) == 1:
        g0.append(f"[{names[plan[0]['layout']]}]null[v0]")
    else:
        uses = {l: [i for i, e in enumerate(plan) if e["layout"] == l] for l in used}
        for l, ids in uses.items():
            if len(ids) > 1:
                g0.append(f"[{names[l]}]split={len(ids)}" + "".join(f"[{names[l]}_{i}]" for i in ids))
            else:
                g0.append(f"[{names[l]}]null[{names[l]}_{ids[0]}]")
        for i, e in enumerate(plan):
            g0.append(f"[{names[e['layout']]}_{i}]trim=start={e['a']:.4f}:end={e['b']:.4f},setpts=PTS-STARTPTS[sc{i}]")
        g0.append("".join(f"[sc{i}]" for i in range(len(plan))) + f"concat=n={len(plan)}:v=1:a=0[v0]")
    vchain = ";".join(g0)

    counts = {l: sum(1 for e in plan if e["layout"] == l) for l in used}
    label = {"track": "speaker tracking", "center": "center crop", "split": "split screen", "screen": "screencast",
             "fit": "fit + blur"}
    how = ", ".join(f"{label[l]}" + (f" ×{counts[l]}" if counts[l] > 1 else "") for l in used)
    if len(plan) > 1:
        how = f"{len(plan)} scenes: " + how
    if s.zoom and ("track" in used or "center" in used):
        how += ", zooms"
    if burned:
        how += ", source already has subtitles" + ("" if explicit_captions else " (ours off)")

    # keep the hook title off the face (single tracked scene): above the head if there's room, else under the chin
    tracked = [e for e in plan if e["layout"] == "track" and e.get("cam") is not None]
    if tracked and used == ["track"]:
        cam = max(tracked, key=lambda e: e["b"] - e["a"])["cam"]
        fy_top = np.median(cam["y"] - 0.75 * cam["h"]) * sc
        fy_bot = np.median(cam["y"] + 0.6 * cam["h"]) * sc
        y0 = min(max(np.median(cam["y"]) * sc - 0.38 * bh, 0), H - bh) if bh < H else 0
        top, bottom = (fy_top - y0) * oh / bh, (fy_bot - y0) * oh / bh
        height = len(wrap(clip.get("title", ""), 20)) * 105 + 40
        if top - 40 < hook_y + height and bottom + 30 + height <= cap_y - 130:
            hook_y = int(bottom + 30)
    if burned and not explicit_captions:
        ws = []                     # don't stack our captions on top of the ones already in the video

    fonts, font_name = font_setup(work, s)
    ass = work / f"{tag}.ass"
    write_ass(ass, ws, tl.length, title, s, ow, oh, font_name, hook_y, cap_y)
    write_srt(out_path.with_suffix(".srt"), ws)

    n = len(pieces)
    g = [vchain]
    g.append(f"[v0]split={n}" + "".join(f"[vs{i}]" for i in range(n)) if n > 1 else "[v0]null[vs0]")
    has_a = info["audio"] and not dub          # cut the source audio along with the video
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
    if dub:                                     # the new voice is already on the output timeline
        inputs += ["-i", str(Path(dub["wav"]).resolve())]
        g.append(f"[{nin}:a]aresample=48000,apad,atrim=0:{tl.length:.3f}[ac]")
        nin += 1
        has_a = True
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

    def render_dub(self, c, dub, lang):
        """Render a dubbed copy of a clip as <clip>.<lang>.mp4 (the original clip is untouched)."""
        s = Settings.from_dict({**self.settings.to_dict(), **c.get("overrides", {})})
        stem = Path(c.get("file") or f"clip{c['id']}.mp4").stem
        name = f"{stem}.{lang}.mp4"
        r = render_clip(Path(self.data["video"]), self.data["info"], c, self.words, self.data.get("language", ""),
                        self.dir / name, self.work, s, c["id"], explicit_captions="captions" in c.get("overrides", {}),
                        dub=dub)
        return {**r, "file": name, "rendered": time.time()}

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
                        self.dir / name, self.work, s, c["id"], explicit_captions="captions" in c.get("overrides", {}))
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
        url = s.ollama.rstrip("/")
        map_cache = p.work / f"map-{hashlib.sha1(json.dumps([s.model, len(words), 1]).encode()).hexdigest()[:10]}.json"
        if map_cache.exists() and not reselect:
            chapters = json.loads(map_cache.read_text())
        else:
            chapters = video_map(sents, s, LANGS.get(language, language))
            map_cache.write_text(json.dumps(chapters, ensure_ascii=False))
        p.data["map"] = chapters
        context = learning_context(out_root)
        sel_key = hashlib.sha1(json.dumps([s.model, s.count, s.min, s.max, s.think, s.prompt, len(words), context,
                                           3]).encode()).hexdigest()[:10]
        sel_cache = p.work / f"picks-{sel_key}.json"
        if sel_cache.exists() and not reselect:
            picks = json.loads(sel_cache.read_text())
            log(f"[2/4] using the AI's previous picks ({sel_cache.name}; --reselect to ask again)", 0.4)
        else:
            chosen = pick_clips(sents, s, LANGS.get(language, language), context)
            picks = [{"i0": sents[c["a"]]["i0"], "i1": sents[c["b"]]["i1"],
                      **{k: c[k] for k in ("score", "scores", "title", "hooks", "platforms", "caption", "hashtags",
                                           "keywords", "topic", "why")}}
                     for c in chosen]
            sel_cache.write_text(json.dumps(picks, indent=1, ensure_ascii=False))
        ollama_unload(url, s.model)   # hand the GPU back
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
        clips.append({**c, **{k: prev[k] for k in ("file", "overrides", "layout", "length", "cut_s", "metrics",
                                                    "rendered") if k in prev}}
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
    write_sheet_csv(p.dir / "clip_sheet.csv", clip_sheet(p))
    return p


# ---------------------------------------------------------------- stitching clips back together

def stitch(p: Project, ids=None, order="source", transition="cut", fade=0.4):
    """Join rendered clips into one video. order: source (as in the original video), score, or as
    given in ids. transition: cut or fade (crossfade). Returns the compilation record."""
    clips = [c for c in p.clips if c.get("file") and (p.dir / c["file"]).exists()]
    if ids:
        by_id = {c["id"]: c for c in clips}
        clips = [by_id[i] for i in ids if i in by_id]
    if order == "source":
        clips.sort(key=lambda c: c["start"])
    elif order == "score":
        clips.sort(key=lambda c: -c.get("score", 0))
    if len(clips) < 2:
        raise SystemExit("Need at least two rendered clips to stitch.")
    ow, oh = ASPECTS.get(p.settings.aspect, ASPECTS["9:16"])
    fps = min(p.data["info"]["fps"], 60.0)
    durs = [float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                  str(p.dir / c["file"])], capture_output=True, text=True).stdout) for c in clips]
    fade = min(fade, min(durs) / 3) if transition == "fade" else 0.0
    has_a = p.data["info"]["audio"]
    log(f"[stitch] joining {len(clips)} clips ({order} order, {transition})...", 0.05)

    inputs, g = [], []
    for i, c in enumerate(clips):
        inputs += ["-hwaccel", "cuda", "-i", str((p.dir / c["file"]).resolve())]
        g.append(f"[{i}:v]scale={ow}:{oh}:force_original_aspect_ratio=decrease:flags=lanczos,"
                 f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps:.4f},format=yuv420p,settb=AVTB[v{i}]")
        if has_a:
            g.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}]")
    n = len(clips)
    if transition == "fade":
        v, a, t = "v0", "a0", 0.0
        for i in range(1, n):
            t += durs[i - 1] - fade
            g.append(f"[{v}][v{i}]xfade=transition=fade:duration={fade:.3f}:offset={t:.3f}[vx{i}]")
            v = f"vx{i}"
            if has_a:
                g.append(f"[{a}][a{i}]acrossfade=d={fade:.3f}[ax{i}]")
                a = f"ax{i}"
        vout, aout = v, a
    else:
        g.append("".join(f"[v{i}]" + (f"[a{i}]" if has_a else "") for i in range(n)) +
                 f"concat=n={n}:v=1:a={1 if has_a else 0}[vc]" + ("[ac]" if has_a else ""))
        vout, aout = "vc", "ac"
    if has_a:
        g.append(f"[{aout}]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]")
    p.work.mkdir(parents=True, exist_ok=True)
    graph = p.work / "stitch.filter"
    graph.write_text(";\n".join(g))
    k = 1 + max((x.get("n", 0) for x in p.data.get("compilations", [])), default=0)
    out = p.dir / f"compilation_{k:02d}.mp4"
    s = p.settings
    enc = s.encoder if s.encoder != "auto" else ("nvenc" if nvenc_ok() else "x264")
    venc = (["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", "19", "-b:v", "0",
             "-maxrate", s.maxrate, "-bufsize", "32M"] if enc == "nvenc"
            else ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-maxrate", s.maxrate, "-bufsize", "32M"])
    cmd = ["ffmpeg", "-v", "error", "-y"] + inputs + ["-filter_complex_script", str(graph.resolve()), "-map", f"[{vout}]"]
    if has_a:
        cmd += ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
    cmd += venc + ["-profile:v", "high", "-pix_fmt", "yuv420p", "-g", str(int(round(fps * 2))),
                   "-movflags", "+faststart", str(out.resolve())]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"ffmpeg failed while stitching:\n{r.stderr[-3000:]}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "1.2", "-i", str(out), "-frames:v", "1", "-q:v", "3",
                    str(out.with_suffix(".jpg"))], check=False)
    length = sum(durs) - fade * (n - 1)
    rec = {"n": k, "file": out.name, "ids": [c["id"] for c in clips], "order": order, "transition": transition,
           "length": round(length, 2), "created": time.time()}
    p.data.setdefault("compilations", []).append(rec)
    p.save()
    log(f"[stitch] {out.name}: {len(clips)} clips, {fmt_ts(length)} long, {time.time() - t0:.0f}s to render", 1.0)
    return rec


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
    p.add_argument("--stitch", choices=["source", "score"],
                   help="also join the clips into one video, in source order or best-first")
    p.add_argument("--transition", choices=["cut", "fade"], default="cut", help="between stitched clips")
    p.add_argument("--schedule", metavar="YYYY-MM-DD", help="write clip_sheet.csv with a posting plan from this date")
    p.add_argument("--platforms", default="instagram", help="for --schedule: e.g. instagram,tiktok,youtube")
    p.add_argument("--per-day", type=int, default=1, help="for --schedule: posts per day")
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
    if args.stitch:
        stitch(proj, order=args.stitch, transition=args.transition)
    if args.schedule:
        plan = schedule(proj, args.schedule, args.per_day, [x.strip() for x in args.platforms.split(",")], args.out)
        write_sheet_csv(proj.dir / "clip_sheet.csv", clip_sheet(proj, plan))
        log(f"clip sheet with posting plan: {proj.dir / 'clip_sheet.csv'}")
    ok = sum(1 for c in proj.clips if c.get("file"))
    log(f"[4/4] done in {time.time() - t0:.0f}s: {ok} shorts in {proj.dir}/ (open {proj.dir / 'index.html'})", 1.0)


if __name__ == "__main__":
    main()
