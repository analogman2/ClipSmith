"""Designer Studio's link to the local ComfyUI (http://127.0.0.1:8188): AI images (Krea 2 Turbo) for B-roll and
actor portraits, and lip-synced talking heads (MiniMax H3 image-to-video driven by a voice track). All on the
GPU; we free ComfyUI's models when done so the card is available for the rest of the pipeline.
"""
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import clipsmith as cs

URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")
COMFY_DIR = Path(os.environ.get("COMFY_DIR", "~/ComfyUI")).expanduser()

IMAGE_MODEL = {"unet": "krea2_turbo_fp8_scaled.safetensors", "clip": "qwen3vl_4b_bf16.safetensors", "clip_type": "krea2",
               "vae": "qwen_image_vae.safetensors", "steps": 8, "cfg": 1.0}
H3 = {"unet": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
      "lora": "urn:air:minimaxh3:lora:huggingface:Kijai%2FMiniMax-H3_comfy@main%2Floras%2Fminimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy.safetensors",
      "clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "vae": "minimax_h3_video_vae_fp16.safetensors",
      "audio_vae": "minimax_h3_audio_vae_fp32.safetensors", "steps": 6, "fps": 24}


def _req(path, body=None, method=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(URL + path, data, {"Content-Type": "application/json"} if data else {}, method=method)
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else {}


def available():
    try:
        _req("/system_stats", timeout=5)
        return True
    except Exception:  # noqa: BLE001
        return False


def upload(path, kind="image"):
    """Copy a local file into ComfyUI's input folder; returns the name its loaders use."""
    path = Path(path)
    name = f"ds_{uuid.uuid4().hex[:8]}_{path.name}"
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: application/octet-stream\r\n\r\n").encode() + path.read_bytes() + \
        f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    r = urllib.request.Request(URL + "/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(r, timeout=120) as resp:
        return json.loads(resp.read())["name"]


def run(prompt, label="ComfyUI", timeout=3600, mem_mb=12000):
    """Queue an API-format graph, wait for it (with pause/stop checkpoints) and return the output files. Holds a
    GPU-broker lease while it runs, so the radio and the other apps make room first."""
    if not available():
        raise SystemExit(f"ComfyUI isn't running at {URL} (needed for AI images and talking heads).")
    with cs.gpu_lease("comfyui", mem_mb, wait_s=120):
        return _run(prompt, label, timeout)


def _run(prompt, label, timeout):
    try:
        pid = _req("/prompt", {"prompt": prompt, "client_id": "designer-studio"})["prompt_id"]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"ComfyUI refused the job: {e.read().decode(errors='replace')[:400]}")
    t0 = time.time()
    try:
        while time.time() - t0 < timeout:
            cs.checkpoint()
            h = _req(f"/history/{pid}")
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("status_str") == "error":
                    msg = next((m[1].get("exception_message", "") for m in st.get("messages", []) if m[0] == "execution_error"), "")
                    raise SystemExit(f"{label} failed in ComfyUI: {msg[:300]}")
                files = []
                for out in h[pid].get("outputs", {}).values():
                    for key in ("images", "videos", "gifs", "audio"):
                        for f in out.get(key, []):
                            files.append(COMFY_DIR / "output" / f.get("subfolder", "") / f["filename"])
                return files
            time.sleep(1.0)
    except BaseException:
        try:                                            # stopped by the user: cancel it in ComfyUI too
            _req("/interrupt", {}, "POST")
            _req("/queue", {"delete": [pid]}, "POST")
        except Exception:  # noqa: BLE001
            pass
        raise
    raise SystemExit(f"{label} took longer than {timeout // 60} minutes.")


def free():
    """Unload ComfyUI's models so the GPU is free for Whisper/Ollama/the voice - unless someone else (Agentora's Image &
    Video Studio, a radio cover) is using ComfyUI right now: then their models stay, and the broker makes room."""
    if cs.card_in_use_by_others():
        return
    try:
        q = _req("/queue", timeout=5)
        if q.get("queue_running") or q.get("queue_pending"):
            return
    except Exception:  # noqa: BLE001
        pass
    try:
        _req("/free", {"unload_models": True, "free_memory": True}, "POST")
    except Exception:  # noqa: BLE001
        pass


def image(prompt, width, height, seed=None, prefix="designer_studio/img"):
    """One Krea 2 Turbo image; returns its path."""
    m = IMAGE_MODEL
    seed = seed if seed is not None else int.from_bytes(os.urandom(6), "big")
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["clip"], "type": m["clip_type"], "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": m["vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": width // 16 * 16, "height": height // 16 * 16, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
                                                   "seed": seed, "steps": m["steps"], "cfg": m["cfg"], "sampler_name": "euler",
                                                   "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }
    files = run(g, "Image")
    return files[0]


def talking_head(portrait, voice_wav, seconds, prompt, width=512, height=896, seed=None, upscale=2.0,
                 prefix="designer_studio/actor"):
    """A lip-synced talking-head clip: MiniMax H3 animates the portrait, driven by the voice track."""
    m = H3
    frames = max(17, int(round(seconds * m["fps"] / 8)) * 8 + 1)
    seed = seed if seed is not None else int.from_bytes(os.urandom(6), "big")
    img, aud = upload(portrait), upload(voice_wav)
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": m["lora"], "strength_model": 1.0}},
        "3": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["2", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["clip"], "type": "minimax", "device": "default"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": m["vae"]}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": m["audio_vae"]}},
        "7": {"class_type": "LoadImage", "inputs": {"image": img}},
        "8": {"class_type": "LoadAudio", "inputs": {"audio": aud}},
        "9": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["4", 0], "vae": ["5", 0], "prompt": prompt, "width": width,
                                                                "height": height, "length": frames, "first_frame": ["7", 0]}},
        "10": {"class_type": "VRGDG_MiniMaxH3AudioDrive", "inputs": {"av_latent": ["9", 1], "source_audio": ["8", 0], "audio_vae": ["6", 0]}},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "12": {"class_type": "BasicScheduler", "inputs": {"model": ["3", 0], "scheduler": "simple", "steps": m["steps"], "denoise": 1.0}},
        "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "14": {"class_type": "BasicGuider", "inputs": {"model": ["3", 0], "conditioning": ["9", 0]}},
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["13", 0], "guider": ["14", 0], "sampler": ["11", 0],
                                                                 "sigmas": ["12", 0], "latent_image": ["10", 0]}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["5", 0]}},
    }
    last = "16"
    if upscale and upscale > 1:
        g["17"] = {"class_type": "RTXVideoSuperResolution", "inputs": {"images": ["16", 0], "scale": upscale, "quality": "ULTRA",
                                                                       "deblur": "OFF"}}
        last = "17"
    g["18"] = {"class_type": "VHS_VideoCombine", "inputs": {"images": [last, 0], "audio": ["8", 0], "frame_rate": float(m["fps"]),
                                                            "loop_count": 0, "filename_prefix": prefix, "format": "video/h264-mp4",
                                                            "pix_fmt": "yuv420p", "crf": 16, "save_metadata": False,
                                                            "trim_to_audio": False, "pingpong": False, "save_output": True}}
    files = run(g, "Talking head", timeout=3600, mem_mb=22000)
    return next(f for f in files if f.suffix.lower() in (".mp4", ".webm", ".mov"))
