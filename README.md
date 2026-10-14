# clipsmith

Our own Opus Clip. Give it a long video; get ready-to-post 1080×1920 Instagram Reels.

1. **Transcribe**: faster-whisper `large-v3-turbo` on the GPU, word-level timestamps.
2. **Pick moments**: a local Ollama model (default `qwen3.6:27b`) reads the transcript like a short-form
   editor. It looks for a hook in the first 3 s, a self-contained idea and a payoff ending. It scores
   each pick 0–100 and writes a title, a caption and hashtags.
3. **Reframe to 9:16**: YuNet face detection plus mouth-motion speaker detection. The crop follows
   whoever is talking, hard-cuts when the speaker changes, and glides smoothly otherwise. Footage
   without faces (screen recordings, B-roll) falls back to "fit" with a blurred background.
4. **Polish**: cuts pauses longer than 0.5 s, adds animated word-by-word captions (Montserrat
   ExtraBold, active word highlighted) and a hook title card, normalizes loudness to −14 LUFS, and
   exports H.264 with NVENC (or x264 if NVENC isn't available).
5. **Gallery**: `index.html` plays every short, with copy-caption buttons, plus `clips.json` and a
   `.txt` per clip.

## Use

```bash
./setup.sh                                   # once
./clipsmith.py podcast.mp4                   # 5 shorts -> clips/podcast/
./clipsmith.py talk.mp4 -n 8 --min 25 --max 50
./clipsmith.py talk.mp4 --dry-run            # just show the picks
./clipsmith.py talk.mp4 --captions green --max-gap 0.35
./clipsmith.py talk.mp4 --clips 1:23-2:05,10:00-10:45   # cut by hand, skip the AI
```

The transcript and the AI's picks are cached in `clips/<name>/.work/`. That means re-rendering with
different caption or layout options is fast. Use `--reselect` to ask the AI again and `--think` for
slower, more deliberate picks. Run `./clipsmith.py -h` for all options.
