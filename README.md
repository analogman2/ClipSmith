# Designer Studio

Our own Opus Clip (the engine and CLI are `clipsmith.py`). Give it a long video (a file or a link) and get ready-to-post shorts for
Instagram Reels, with a web app to review, edit and re-render them. Everything runs locally and
**GPU only**: if Whisper or the picker model doesn't fit in VRAM, Designer Studio stops and tells you what
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
- **Stitch clips**: join all clips, or the ones you tick, into one video. Order them as in the
  original, best first, or in the order you ticked them, with hard cuts or crossfades.
- **Download**: MP4 and SRT, plus copy the caption and hashtags.

Jobs run one at a time, because they share the GPU. The progress bar at the top shows the current one.

### Publishing tools
- **YouTube Studio** (timeline toolbar): 10 titles with a refine chat, a description with chapter
  timestamps from the video map, tags, a pinned comment, and thumbnails built from frames where the
  speaker's eyes are open and the face is expressive.
- **Instagram kit** (inspector tab): 5 caption options with a refine chat, broad/niche/community
  hashtag sets, alt text, a first comment, and a Reels cover with the title inside the 3:4 area the
  profile grid shows.
- **Safe zones** (viewer): shows where the Reels/TikTok buttons and caption cover the video.
- **Dubbing** (Clip tab): translates a clip and re-voices it in the speaker's own cloned voice
  (Qwen3-TTS Base), with new captions and hook. English, Spanish, Portuguese, French, German,
  Italian, Japanese, Korean, Chinese, Russian.

## What it does

1. **Transcribes**: faster-whisper `large-v3-turbo` on the GPU, with word-level timestamps and
   language detection.
2. **Picks moments**: Ollama (default `qwen3.6-agent`, the Qwen 3.6 27B that Agentora's Hermes and radio also use, checked to fit fully in VRAM) acts as a
   short-form editor. It looks for a hook, a self-contained idea and a payoff. Each pick gets a score
   with a breakdown, a title, a caption, hashtags and keywords, written in the video's own language.
3. **Reframes**: detects camera cuts and lays out each scene separately. A local vision model
   (`qwen3.5-agent-fast`, Qwen 3.5 9B) says what each scene shows (one person, two people, a screen share) and spots
   burned-in subtitles, so ours stay off. YuNet face detection with mouth-motion speaker detection follows whoever is
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
./clipsmith.py talk.mp4 --stitch source --transition fade  # also join them into one video
```

Run `./clipsmith.py -h` for everything. Transcripts and AI picks are cached in
`clips/<name>/.work/`, so restyling is fast.

Dubbing runs the voice in the speech environment (`~/venv-speech`, set `DUB_PYTHON` to change it)
under the GPU broker. It needs the `Qwen/Qwen3-TTS-12Hz-1.7B-Base` model (about 4 GB in the
Hugging Face cache).

## Not included (needs outside services)

AI B-roll (needs a stock footage API such as Pexels), auto-posting to Instagram (needs a Meta
Business account and Graph API token), and emoji in captions (libass can't draw color emoji).

## Sharing the GPU with Agentora

The card is shared with Agentora (radio, Image & Video Studio, Hermes), through Agentora's GPU broker (`127.0.0.1:7890`):

- **Ollama calls** go through the broker's proxy (`http://127.0.0.1:11435`, the default `ollama` setting), sent as client `designer-studio`.
  - The proxy takes a lease for each call and shares the model Hermes already has loaded, so there's one copy on the card.
  - A smaller `num_ctx` doesn't reload it. "Unload" requests leave it loaded while someone else is using it.
- **Whisper** (`designer-studio-whisper`, ~4 GB) and every **ComfyUI job** (`designer-studio-comfyui`: 12 GB for an image, 22 GB for a talking head) hold a lease while they run (`clipsmith.gpu_lease`).
  - A person is waiting, so the radio steps aside after its current take.
- **The dub voice** (`dub_worker.py`, `designer-studio-dub`) holds its own lease.
- **`comfy.free()`** leaves ComfyUI's models alone while Agentora's Studio or a radio cover is using them.
- **Old saved settings** naming `qwen3.6:27b`, `qwen3.5:9b` or Ollama on port 11434 are mapped to the shared names and the proxy.
- **Without the broker** everything still runs, just without the coordination.
