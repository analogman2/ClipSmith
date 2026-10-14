"""Designer Studio dubbing voice: Qwen3-TTS 1.7B Base clones the speaker from a short reference clip and
speaks the translated lines. Runs in the speech venv (~/venv-speech, which has torch + faster-qwen3-tts),
on the GPU only, under a GPU-broker lease like Agentora's speech service.

    ~/venv-speech/bin/python dub_worker.py job.json      # writes line_<i>.wav next to job.json
job.json: {"ref_audio", "ref_text", "language", "lines": ["...", ...], "out_dir"}
"""
import json
import os
import sys
import time
import urllib.request

MODEL = os.environ.get("DUB_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
BROKER = os.environ.get("GPU_BROKER", "http://127.0.0.1:7890")


def broker(method, path, body=None, timeout=30):
    r = urllib.request.Request(BROKER + path, json.dumps(body).encode() if body is not None else None,
                               {"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read() or b"{}")


def main():
    job = json.load(open(sys.argv[1]))
    import numpy as np
    import soundfile as sf
    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA isn't available: the dubbing voice only runs on the GPU.")
    lease = None
    try:
        lease = broker("POST", "/lease", {"client": "designer-studio-dub", "kind": "batch", "mem_mb": 6000,
                                          "wait_s": 120, "ttl_s": 900}, 140).get("id")
    except Exception as e:  # noqa: BLE001
        print(f"GPU broker not reachable ({e}); loading anyway", file=sys.stderr)
    try:
        from faster_qwen3_tts import FasterQwen3TTS
        t0 = time.time()
        m = FasterQwen3TTS.from_pretrained(MODEL, device="cuda")
        print(f"voice model loaded in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
        out = []
        for i, text in enumerate(job["lines"]):
            torch.manual_seed(1234 + i)
            wavs, sr = m.generate_voice_clone(text=text, language=job["language"], ref_audio=job["ref_audio"],
                                              ref_text=job["ref_text"], append_silence=False)
            audio = np.concatenate([np.asarray(w, np.float32).reshape(-1) for w in wavs]) if wavs else np.zeros(1, np.float32)
            f = os.path.join(job["out_dir"], f"line_{i:03d}.wav")
            sf.write(f, audio, sr)
            out.append({"file": f, "dur": len(audio) / sr})
            print(f"line {i + 1}/{len(job['lines'])}", file=sys.stderr, flush=True)
            if lease:
                try:
                    broker("POST", f"/lease/{lease}/renew", {"ttl_s": 900}, 5)
                except Exception:  # noqa: BLE001
                    pass
        json.dump(out, open(os.path.join(job["out_dir"], "lines.json"), "w"))
    finally:
        if lease:
            try:
                broker("DELETE", f"/lease/{lease}", timeout=5)
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    main()
