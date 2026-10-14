"""Designer Studio Clip Generator: the guided path. Two ways in:

From your video: analyze -> pick moments -> hook -> look (captions, color grade) -> voice (keep or dub) ->
  AI B-roll -> render -> export or post.
No video (AI Shorts): describe it (or a web page) -> script (hook, problem, solution, call to action) ->
  AI actor portrait -> designed voice -> lip-synced talking head (MiniMax H3) -> then the same look,
  B-roll, render and export steps.

Everything runs locally: Ollama writes, ComfyUI draws and animates, Qwen3-TTS speaks, ffmpeg renders.
"""
import html
import json
import re
import shutil
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

import clipsmith as cs
import comfy
import dub
import studio

HERE = Path(__file__).resolve().parent
AISHORTS = HERE / "aishorts"


# ---------------------------------------------------------------- AI B-roll for a clip

BROLL_PROMPT = """You plan B-roll cutaways for a vertical short. You get the spoken lines with their times \
(seconds in the finished short). Pick {n} moments where a cutaway image would add meaning (a concrete thing, \
place, action or metaphor that is mentioned) - never the first 2 seconds (that is the hook) and not the last \
second. For each give: line (the index of the line), and prompt: an English prompt for a photorealistic, \
cinematic vertical image (9:16) of that thing - no people talking to camera, no text, no logos, natural light, \
shallow depth of field."""
BROLL_SCHEMA = {"type": "object", "properties": {"cuts": {"type": "array", "items": {
    "type": "object", "properties": {"line": {"type": "integer"}, "prompt": {"type": "string"}},
    "required": ["line", "prompt"]}}}, "required": ["cuts"]}


def broll_suggest(p: cs.Project, c, n=3):
    """Pick moments for AI B-roll and draw them (Krea 2 Turbo via ComfyUI)."""
    s = studio.Settings_for(p, c)
    tl, ws = dub.output_words(p, c, s)
    lines = dub.group_lines(ws)
    if not lines:
        raise SystemExit("This clip has no speech to plan B-roll around.")
    cs.log("[broll] choosing moments for B-roll...", 0.05)
    d = studio.llm_json(s, BROLL_PROMPT.format(n=n), "\n".join(f"[{i}] ({l['s']:.1f}s) {l['text']}" for i, l in enumerate(lines)),
                        BROLL_SCHEMA, num_ctx=8192, temperature=0.5)
    cs.ollama_unload(s.ollama.rstrip("/"), s.model)
    cuts, used = [], set()
    for x in d["cuts"]:
        i = int(x["line"])
        if (0 <= i < len(lines) and i not in used and lines[i]["s"] >= 1.8 and lines[i]["s"] < tl.length - 1.5
                and all(abs(lines[i]["s"] - u["start"]) >= 3.5 for u in cuts)):    # room between cutaways
            used.add(i)
            cuts.append({"start": round(lines[i]["s"] + 0.15, 2), "dur": 2.4, "prompt": x["prompt"].strip(),
                         "line": lines[i]["text"], "on": True})
    cuts = sorted(cuts, key=lambda b: b["start"])[:n]
    out = p.dir / "studio"
    out.mkdir(exist_ok=True)
    for k, b in enumerate(cuts):
        cs.log(f"[broll] drawing image {k + 1}/{len(cuts)}...", 0.15 + 0.8 * k / max(1, len(cuts)))
        b["file"] = draw_broll(b["prompt"], out / f"broll_{c['id']}_{k}_{uuid.uuid4().hex[:6]}.png", s.broll_style)
    comfy.free()
    c["broll"] = cuts
    p.save()
    return cuts


BROLL_STYLES = {"cinematic": "cinematic photo, photorealistic, shallow depth of field",
                "lifestyle": "bright natural lifestyle photo, airy, soft daylight",
                "minimal": "minimalist clean photo, soft muted colors, lots of negative space",
                "illustrated": "flat modern illustration, clean shapes, soft colors"}


def draw_broll(prompt, dest, style="cinematic"):
    src = comfy.image(f"{prompt}, {BROLL_STYLES.get(style, BROLL_STYLES['cinematic'])}, vertical 9:16, no text", 1088, 1920,
                      prefix="designer_studio/broll")
    shutil.copy(src, dest)
    return str(dest)


def broll_redraw(p: cs.Project, c, k, prompt=None):
    b = c["broll"][k]
    if prompt:
        b["prompt"] = prompt
    cs.log("[broll] drawing a new image...", 0.2)
    b["file"] = draw_broll(b["prompt"], p.dir / "studio" / f"broll_{c['id']}_{k}_{uuid.uuid4().hex[:6]}.png",
                           studio.Settings_for(p, c).broll_style)
    comfy.free()
    p.save()
    return b


# ---------------------------------------------------------------- AI color grade

LOOK_SCHEMA = {"type": "object", "properties": {"look": {"type": "string", "enum": list(cs.LOOKS)},
                                                "why": {"type": "string"}}, "required": ["look", "why"]}


def pick_look(p: cs.Project, c):
    """The vision model looks at a frame and the topic and picks a color grade from the fixed set."""
    import base64
    from io import BytesIO
    s = studio.Settings_for(p, c)
    img = studio.grab(Path(p.data["video"]), c["start"] + min(2, (c["end"] - c["start"]) / 3), 512)
    buf = BytesIO()
    img.save(buf, "JPEG", quality=85)
    url = s.ollama.rstrip("/")
    cs.ensure_model_on_gpu(url, s.vision_model, 4096)
    r = cs._post(url + "/api/chat", {"model": s.vision_model, "stream": False, "format": LOOK_SCHEMA, "think": False,
                                     "options": {"temperature": 0.2, "num_ctx": 4096},
                                     "messages": [{"role": "user", "images": [base64.b64encode(buf.getvalue()).decode()],
                                                   "content": f"This frame is from a short video about: {c.get('topic') or c.get('title')}. "
                                                              f"Pick the color grade that would make it look best on Instagram, from: "
                                                              f"{', '.join(cs.LOOKS)}. Keep skin tones natural. why: one short sentence (max 15 words)."}]}, 120)
    return json.loads(r["message"]["content"])


# ---------------------------------------------------------------- No video: AI Shorts

def fetch_page(url, limit=9000):
    """Readable text from a web page (for writing a script about a product or site)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (DesignerStudio)"})
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read(3_000_000).decode(errors="replace")
    raw = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", raw)
    title = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    text = re.sub(r"\s+", " ", text).strip()
    return ((html.unescape(title.group(1)).strip() + "\n") if title else "") + text[:limit]


SCRIPT_PROMPT = """You write scripts for vertical short-form videos (Reels, TikTok, Shorts) spoken by one \
on-camera presenter. Language: {language}. Target length: about {seconds} seconds of speech (about \
{words} words in total). Structure: hook (1 line, stops the scroll in 2 seconds), problem (1-2 lines), \
solution (2-4 lines, concrete), cta (1 line). Write the way people talk: short sentences, no hashtags, no \
emojis, no stage directions. Each line is one breath (max ~16 words).
Also give: title (on-screen hook text, max 8 words), broll (3 English prompts for photorealistic vertical \
cutaway images that illustrate the solution; no text, no people talking to camera), caption (Instagram \
caption with a call to action) and hashtags (6-8, no # sign)."""
SCRIPT_SCHEMA = {"type": "object", "properties": {
    "title": {"type": "string"},
    "segments": {"type": "array", "items": {"type": "object", "properties": {
        "part": {"type": "string", "enum": ["hook", "problem", "solution", "cta"]},
        "lines": {"type": "array", "items": {"type": "string"}}}, "required": ["part", "lines"]}},
    "broll": {"type": "array", "items": {"type": "string"}}, "caption": {"type": "string"},
    "hashtags": {"type": "array", "items": {"type": "string"}}},
    "required": ["title", "segments", "broll", "caption", "hashtags"]}


class Short:
    """State of one no-video short (aishorts/<id>/state.json)."""

    def __init__(self, sid=None):
        self.id = sid or uuid.uuid4().hex[:10]
        self.dir = AISHORTS / self.id
        self.file = self.dir / "state.json"
        self.data = json.loads(self.file.read_text()) if self.file.exists() else {"id": self.id, "created": time.time()}

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))
        tmp.replace(self.file)


def settings():
    return cs.Settings.from_dict({**cs.Settings().to_dict(), **cs.load_brand(HERE / "brand.json")})


def write_script(sh: Short):
    d = sh.data
    lang = d.get("language", "en")
    s = settings()
    brief = d.get("description", "")
    if d.get("url"):
        cs.log("[script] reading the web page...", 0.05)
        try:
            brief = f"Web page ({d['url']}):\n{fetch_page(d['url'])}\n\nNotes from the creator: {brief}"
        except Exception as e:  # noqa: BLE001
            cs.log(f"[script] couldn't read the page ({str(e)[:80]}); using the description only")
    if d.get("audience"):
        brief += f"\nAudience: {d['audience']}"
    seconds = int(d.get("seconds", 30))
    cs.log("[script] writing the script...", 0.3)
    out = studio.llm_json(s, SCRIPT_PROMPT.format(language=dub.DUB_LANGS.get(lang, "English"), seconds=seconds,
                                                  words=int(seconds * 2.4)), studio.profile_text() + brief, SCRIPT_SCHEMA,
                          num_ctx=16384, temperature=0.7)
    cs.ollama_unload(s.ollama.rstrip("/"), s.model)
    d["script"] = out
    sh.save()
    return out


def portraits(sh: Short, n=4):
    d = sh.data
    look = d.get("actor") or "a friendly, confident person in their 30s"
    out = []
    for k in range(n):
        cs.log(f"[actor] drawing actor {k + 1}/{n}...", 0.1 + 0.85 * k / n)
        src = comfy.image(f"photorealistic vertical portrait photo of {look}, looking straight into the camera, "
                          f"mouth closed, relaxed natural expression, head and shoulders centered in the upper half, "
                          f"clean bright modern background, soft natural light, sharp focus, full-frame photo with no "
                          f"phone, screen or device frame visible", 1088, 1920,
                          prefix="designer_studio/actor_portrait")
        dest = sh.dir / f"portrait_{k}_{uuid.uuid4().hex[:6]}.png"
        sh.dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        out.append(str(dest))
    comfy.free()
    d["portraits"] = out
    d.setdefault("portrait", out[0])
    sh.save()
    return out


def all_lines(sh: Short):
    return [ln.strip() for seg in sh.data["script"]["segments"] for ln in seg["lines"] if ln.strip()]


def record_voice(sh: Short):
    d = sh.data
    lines = all_lines(sh)
    work = sh.dir / "voice"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    desc = d.get("voice") or "A warm, confident, natural voice with a friendly conversational pace."
    cs.log(f"[voice] recording {len(lines)} lines...", 0.1)
    spoken = dub.speak({"design": desc, "language": dub.DUB_LANGS.get(d.get("language", "en"), "English"), "lines": lines,
                        "out_dir": str(work)}, work)
    # one track, a short breath between lines
    parts = []
    for sp in spoken:
        parts.append(dub.load_pcm(sp["file"]))
        parts.append(__import__("numpy").zeros(int(0.18 * dub.SR), "float32"))
    import numpy as np
    pcm = np.concatenate(parts)
    pcm = pcm / max(1e-6, float(np.abs(pcm).max())) * 0.9
    wav = sh.dir / "voice.wav"
    import wave
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(dub.SR)
        w.writeframes((pcm * 32767).astype("<i2").tobytes())
    d["voice_wav"] = str(wav)
    d["line_durs"] = [round(sp["dur"] + 0.18, 3) for sp in spoken]
    d["voice_len"] = round(len(pcm) / dub.SR, 2)
    sh.save()
    return d["voice_len"]


def chunks(sh: Short, limit=8.0):
    """Group lines into talking-head chunks of at most ~8 s (one MiniMax H3 run each)."""
    out, cur, t = [], [], 0.0
    pos = 0.0
    for dur in sh.data["line_durs"]:
        if cur and t + dur > limit:
            out.append((pos - t, t))
            cur, t = [], 0.0
        cur.append(dur)
        t += dur
        pos += dur
    if cur:
        out.append((pos - t, t))
    return out


def make_video(sh: Short):
    """Talking-head video for the whole script, then a regular project from it."""
    d = sh.data
    parts = chunks(sh)
    vids = []
    actor = d.get("actor") or "a person"
    for k, (a, dur) in enumerate(parts):
        cs.log(f"[video] animating part {k + 1} of {len(parts)} (about {int(dur)}s)...", 0.05 + 0.8 * k / len(parts))
        seg = sh.dir / f"voice_{k}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-t", f"{dur:.3f}", "-i", d["voice_wav"], str(seg)], check=True)
        v = comfy.talking_head(d["portrait"], seg, dur, f"{actor} talks directly to the camera with natural expressions and "
                                                        f"subtle head movement, speaking clearly, smartphone selfie video, "
                                                        f"steady camera, same person and background throughout.")
        dest = sh.dir / f"part_{k}.mp4"
        shutil.copy(v, dest)
        vids.append(dest)
    comfy.free()
    cs.log("[video] putting it together...", 0.88)
    lst = sh.dir / "parts.txt"
    lst.write_text("".join(f"file '{v.name}'\n" for v in vids))
    raw = sh.dir / "video_only.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-an", "-c:v", "h264_nvenc",
                    "-cq", "17", "-pix_fmt", "yuv420p", str(raw)], check=True, cwd=sh.dir)
    title = cs.slug(d.get("script", {}).get("title") or "ai short")
    final = sh.dir / f"{title}-{sh.id[:4]}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-i", d["voice_wav"], "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(final)], check=True)
    d["video"] = str(final)
    sh.save()
    return str(final)


def make_project(sh: Short, root, base=None, suffix=""):
    """Turn the AI video into a normal project (one clip covering it all), carrying over the script's
    hook, caption and B-roll ideas. base: the user's default settings."""
    d = sh.data
    s = cs.Settings.from_dict(base) if base else settings()
    s.language = d.get("language", "en")
    info = cs.probe(d["video"])
    p = cs.create_project(d["video"], root, s, manual=[(0.0, info["duration"])], suffix=suffix)
    if d.get("owner"):
        p.data["owner"] = d["owner"]
    c = p.clips[0]
    sc = d.get("script", {})
    c.update(title=sc.get("title") or c["title"], caption=sc.get("caption", ""), hashtags=sc.get("hashtags", []),
             hooks=[sc.get("title")] if sc.get("title") else [], why="AI short", topic=d.get("topic", ""))
    c["platforms"] = {"instagram": {"caption": c["caption"], "hashtags": c["hashtags"]}}
    c["broll_ideas"] = sc.get("broll", [])
    p.save()
    d["project"] = p.dir.name
    sh.save()
    return p.dir.name


def broll_from_ideas(p: cs.Project, c):
    """B-roll for an AI short: the script's own ideas, spread over the solution part."""
    ideas = c.get("broll_ideas") or []
    if not ideas:
        return broll_suggest(p, c)
    s = studio.Settings_for(p, c)
    tl, ws = dub.output_words(p, c, s)
    L = tl.length
    slots = [L * f for f in (0.35, 0.55, 0.75)][:len(ideas)]
    out = p.dir / "studio"
    out.mkdir(exist_ok=True)
    cuts = []
    for k, (t, prompt) in enumerate(zip(slots, ideas)):
        cs.log(f"[broll] drawing image {k + 1}/{len(slots)}...", 0.1 + 0.85 * k / len(slots))
        cuts.append({"start": round(t, 2), "dur": 2.4, "prompt": prompt, "line": "", "on": True,
                     "file": draw_broll(prompt, out / f"broll_{c['id']}_{k}_{uuid.uuid4().hex[:6]}.png", s.broll_style)})
    comfy.free()
    c["broll"] = cuts
    p.save()
    return cuts
