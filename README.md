# clipsmith

Our own Opus Clip. Give it a long video (a file or a link) and get ready-to-post shorts for
Instagram Reels, with a web app to review, edit and re-render them. Everything runs locally and
**GPU only**: if Whisper or the picker model doesn't fit in VRAM, clipsmith stops and tells you what
to free. It never falls back to the CPU.

## Web app

```bash
./setup.sh      # once
./serve.sh      # http://127.0.0.1:8765
```

- **New clips**: pick a video from `~/Videos`, upload one, or paste a YouTube or other link. You can
  say what the clips should be about, and choose how many, how long, the aspect ratio, caption
  style, layout, colors, logo and music.
- **Review**: clips are sorted by virality score, with a breakdown (hook, flow, value, trend) and the
  AI's reason for each pick.
- **Edit**: change the title, caption, hashtags and highlighted keywords, nudge the start/end, and
  set the caption style, layout or aspect per clip. Then **Save & render**.
- **Clip from transcript**: select sentences and make your own clip.
- **Find new moments**: ask the AI again with a new prompt, count or length.
- **Restyle all**: re-render every clip with new captions, aspect, colors, logo or music.
- **Download**: MP4 and SRT, plus copy the caption and hashtags.

Jobs run one at a time, because they share the GPU. The progress bar at the top shows the current one.

## What it does

1. **Transcribes**: faster-whisper `large-v3-turbo` on the GPU, with word-level timestamps and
   language detection.
2. **Picks moments**: Ollama (default `qwen3.6:27b`, checked to fit fully in VRAM) acts as a
   short-form editor. It looks for a hook, a self-contained idea and a payoff. Each pick gets a score
   with a breakdown, a title, a caption, hashtags and keywords, written in the video's own language.
3. **Reframes**: YuNet face detection with mouth-motion speaker detection follows whoever is
   talking and hard-cuts when the speaker changes. Two people who are far apart get a **split
   screen**. Footage without faces gets a blurred-background **fit**. Outputs 9:16, 4:5 or 1:1.
4. **Edits**: cuts filler words (um/uh/hã…) and pauses. Adds alternating **punch-in zooms** on
   jump cuts and a quick **zoom pop** on keywords.
5. **Captions**: animated templates `bold`, `green`, `karaoke`, `pop` (one word at a time) and
   `clean`. The active word is highlighted and AI keywords get a second color. Also adds a hook title
   that avoids the face, and a **progress bar**.
6. **Brand and audio**: logo overlay, your own caption font, background music that ducks under the
   voice, −14 LUFS loudness.
7. **Export**: H.264 via NVENC (1080 wide, ≤16 Mbit/s), a `.srt` and a `.txt` (caption and
   hashtags) per clip, and `project.json`.

Put defaults (font, colors, logo, music…) in `brand.json`. Any field of `Settings` in clipsmith.py
works, for example `{"highlight": "#FF3EA5", "logo": "~/brand/logo.png", "captions": "pop"}`.

## CLI

```bash
./clipsmith.py talk.mp4                                  # 5 shorts -> clips/talk/
./clipsmith.py "https://youtu.be/..." -n 8 --max 45
./clipsmith.py talk.mp4 --prompt "the pricing story" --captions pop --aspect 4:5
./clipsmith.py talk.mp4 --clips 1:23-2:05,10:00-10:45   # cut by hand, skip the AI
./clipsmith.py talk.mp4 --dry-run                        # just show the picks
```

Run `./clipsmith.py -h` for everything. Transcripts and AI picks are cached in
`clips/<name>/.work/`, so restyling is fast.

## Not included (needs outside services)

AI B-roll (needs a stock footage API such as Pexels), auto-posting to Instagram (needs a Meta
Business account and Graph API token), and emoji in captions (libass can't draw color emoji).
