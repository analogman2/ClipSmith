"""Designer Studio dubbing: translate a clip and re-voice it in the speaker's own (cloned) voice.

transcript lines -> local LLM translates them (similar length, natural spoken style) -> Qwen3-TTS Base
clones the voice from a clean stretch of the clip and speaks each line -> lines are fitted to the original
timing (gently sped up when longer) -> Whisper re-times the captions on the new audio -> the clip is
rendered again with the new voice, captions and hook. GPU only; models are loaded one at a time.
"""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

import clipsmith as cs
import studio

SPEECH_PY = Path(os.environ.get("DUB_PYTHON", "~/venv-speech/bin/python")).expanduser()
WORKER = Path(__file__).resolve().parent / "dub_worker.py"
DUB_LANGS = {"en": "English", "es": "Spanish", "pt": "Portuguese", "fr": "French", "de": "German", "it": "Italian",
             "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "ru": "Russian"}
SR = 48000

TRANSLATE_PROMPT = """You translate a short video for dubbing into {target}. You get the hook title and the \
spoken lines in order. Return:
- lines: exactly one translation per input line, same order and count. Natural, spoken {target} as a native \
creator would say it; keep the meaning, tone and energy. Each translation should take about as long to say as \
the original (similar syllable count), so the voice fits the same timing. No notes, no quotes.
- title: the hook title in {target}, max 8 words, punchy.
- keywords: 3-6 single words copied from your translated lines that carry the punch."""
TRANSLATE_SCHEMA = {"type": "object", "properties": {"lines": {"type": "array", "items": {"type": "string"}},
                                                     "title": {"type": "string"},
                                                     "keywords": {"type": "array", "items": {"type": "string"}}},
                    "required": ["lines", "title", "keywords"]}


def output_words(p, c, s):
    """The clip's spoken words on the output timeline (after pause/filler cuts), like the renderer sees them."""
    words, language = p.words, p.data.get("language", "")
    pieces, kept = cs.keep_intervals(words, c["i0"], c["i1"], c["start"], c["end"], s.max_gap, s.fillers, language)
    tl = cs.Timeline(pieces)
    return tl, [{"w": words[i]["w"], "s": tl(words[i]["s"] - c["start"]), "e": tl(words[i]["e"] - c["start"]),
                 "src_s": words[i]["s"], "src_e": words[i]["e"]} for i in kept]


def group_lines(ws):
    lines, cur = [], []
    for i, w in enumerate(ws):
        cur.append(w)
        nxt = ws[i + 1] if i + 1 < len(ws) else None
        if not nxt or w["w"][-1] in ".?!" or nxt["s"] - w["e"] > 0.45 or len(cur) >= 22:
            lines.append({"s": cur[0]["s"], "e": cur[-1]["e"], "src_s": cur[0]["src_s"], "src_e": cur[-1]["src_e"],
                          "text": " ".join(x["w"] for x in cur)})
            cur = []
    return lines


def reference(p, lines, work):
    """A clean 6-12 s stretch of the speaker (consecutive lines in the source) and what they say in it."""
    best = None
    for i in range(len(lines)):
        j, text = i, []
        while j < len(lines) and lines[j]["src_e"] - lines[i]["src_s"] <= 12:
            text.append(lines[j]["text"])
            j += 1
        if j == i:
            continue
        span = lines[j - 1]["src_e"] - lines[i]["src_s"]
        if best is None or abs(span - 9) < abs(best[1] - 9):
            best = (lines[i]["src_s"], span, " ".join(text))
    a, span, text = best
    ref = work / "ref.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0, a - 0.1):.3f}", "-t", f"{span + 0.25:.3f}",
                    "-i", p.data["video"], "-vn", "-ac", "1", "-ar", "24000", str(ref)], check=True)
    return ref, text


def speak(job, work):
    """Run the cloning voice in the speech venv (GPU only)."""
    if not SPEECH_PY.exists():
        raise SystemExit(f"Dubbing needs the speech environment at {SPEECH_PY} (set DUB_PYTHON).")
    f = work / "job.json"
    f.write_text(json.dumps(job, ensure_ascii=False))
    proc = subprocess.Popen([str(SPEECH_PY), str(WORKER), str(f)], stderr=subprocess.PIPE, text=True)
    for line in proc.stderr:
        line = line.strip()
        if line.startswith("line ") or line.startswith("voice model"):
            cs.log(f"[dub] {line}", None)
    if proc.wait():
        raise SystemExit("The dubbing voice failed (see the log above).")
    return json.loads((work / "lines.json").read_text())


def load_pcm(path, tempo=1.0):
    af = f"atempo={tempo:.4f}," if abs(tempo - 1) > 0.01 else ""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-af", f"{af}aresample={SR}", "-ac", "1", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def fit_track(lines, spoken, length, work):
    """Place each spoken line at its original start; speed it up (max 1.35x) when it's longer than its slot."""
    buf = np.zeros(int(length * SR) + SR, np.float32)
    cursor, squeezed = 0.0, 0
    for k, (ln, sp) in enumerate(zip(lines, spoken)):
        slot_end = lines[k + 1]["s"] if k + 1 < len(lines) else length
        slot = max(0.3, slot_end - ln["s"])
        tempo = min(1.35, max(1.0, sp["dur"] / slot))
        squeezed += tempo > 1.01
        pcm = load_pcm(sp["file"], tempo)
        start = max(ln["s"] - 0.03, cursor)
        i0 = int(start * SR)
        i1 = min(len(buf), i0 + len(pcm))
        buf[i0:i1] += pcm[:i1 - i0]
        cursor = start + len(pcm) / SR + 0.04
    buf = buf[:int(length * SR)]
    peak = float(np.abs(buf).max() or 1)
    buf = buf / peak * 0.89
    out = work / "dub.wav"
    import wave
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((buf * 32767).astype("<i2").tobytes())
    overflow = max(0.0, cursor - length)
    return out, squeezed, overflow


def retime_words(wav, lang, s):
    """Word timings of the new voice (Whisper on the GPU), for the captions."""
    cs.load_cuda12_libs()
    from faster_whisper import WhisperModel
    pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    with cs.gpu_lease("whisper", 4000):                 # the card is shared: hold it while Whisper runs
        try:
            model = WhisperModel(s.whisper, device="cuda", compute_type="float16")
        except Exception as e:  # noqa: BLE001
            raise cs.GpuError(f"Whisper can't load on the GPU ({str(e)[:100]}). clipsmith never uses the CPU.")
        segs, _ = model.transcribe(np.frombuffer(pcm, np.float32), language=lang, word_timestamps=True, vad_filter=False)
        words = [{"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)} for seg in segs for w in (seg.words or [])
                 if w.word.strip()]
        del model
    return words


def dub_clip(p: cs.Project, c, lang):
    if lang not in DUB_LANGS:
        raise SystemExit(f"Dubbing supports: {', '.join(DUB_LANGS.values())}")
    if c.get("i0") is None:
        raise SystemExit("This clip has no speech to dub.")
    if lang == p.data.get("language"):
        raise SystemExit("That's already the clip's language.")
    s = studio.Settings_for(p, c)
    target = DUB_LANGS[lang]
    work = p.work / f"dub-{c['id']}-{lang}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    t0 = time.time()

    tl, ws = output_words(p, c, s)
    lines = group_lines(ws)
    cs.log(f"[dub] translating {len(lines)} lines into {target}...", 0.1)
    d = studio.llm_json(s, TRANSLATE_PROMPT.format(target=target),
                        json.dumps({"title": c.get("title", ""), "lines": [l["text"] for l in lines]}, ensure_ascii=False),
                        TRANSLATE_SCHEMA, num_ctx=8192, temperature=0.3)
    cs.ollama_unload(s.ollama.rstrip("/"), s.model)           # make room for the voice
    tr = [x.strip() for x in d["lines"]]
    if len(tr) != len(lines):                                 # keep the timing even if the model merged lines
        tr = (tr + [""] * len(lines))[:len(lines)]
    keep = [(l, t) for l, t in zip(lines, tr) if t]
    lines, tr = [l for l, _ in keep], [t for _, t in keep]

    ref, ref_text = reference(p, lines, work)
    cs.log(f"[dub] cloning the voice and speaking {len(tr)} lines in {target}...", 0.3)
    spoken = speak({"ref_audio": str(ref), "ref_text": ref_text, "language": target, "lines": tr,
                    "out_dir": str(work)}, work)
    wav, squeezed, overflow = fit_track(lines, spoken, tl.length, work)
    cs.log(f"[dub] timing: {squeezed} line(s) sped up slightly" + (f", {overflow:.1f}s cut at the end" if overflow > 0.2 else ""), 0.65)
    cs.log("[dub] timing the captions to the new voice...", 0.7)
    words = retime_words(wav, lang, s)
    kws = {cs.norm_word(k) for k in d["keywords"]} - {""}
    dub = {"wav": str(wav), "title": d["title"].strip() or c.get("title", ""),
           "words": [{**w, "kw": cs.norm_word(w["w"]) in kws} for w in words]}
    cs.log("[dub] rendering...", 0.8)
    r = p.render_dub(c, dub, lang)
    r.update(lines=[{"from": l["text"], "to": t} for l, t in zip(lines, tr)], title=dub["title"],
             took_s=round(time.time() - t0))
    c.setdefault("dubs", {})[lang] = r
    p.save()
    cs.log(f"[dub] done: {r['file']} ({r['took_s']}s)", 1.0)
    return r
