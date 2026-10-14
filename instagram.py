"""Posting a Reel to Instagram through the official Instagram Graph API (no third-party service).

Needs an Instagram Creator or Business account and an access token you create in Meta's developer
dashboard (Instagram API with Instagram Login gives tokens starting "IG"; with Facebook Login, "EAA").
The token is stored only in instagram.json next to this file (never sent to the browser).

Flow: create a REELS container with resumable upload -> upload the MP4 bytes straight from this machine
(rupload.facebook.com) -> wait until Instagram has processed it -> publish -> fetch the permalink.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import clipsmith as cs

CONFIG = Path(__file__).resolve().parent / "instagram.json"
VERSION = "v21.0"


def load():
    try:
        return json.loads(CONFIG.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def graph_base(cfg):
    if cfg.get("graph_base"):                         # for testing against a local stand-in
        return cfg["graph_base"].rstrip("/")
    return "https://graph.facebook.com" if cfg.get("token", "").startswith("EAA") else "https://graph.instagram.com"


def _call(method, url, params=None, data=None, headers=None, timeout=60):
    if params:
        url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            msg = json.loads(body).get("error", {}).get("message", body)
        except json.JSONDecodeError:
            msg = body
        raise SystemExit(f"Instagram said: {msg[:300]}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Couldn't reach Instagram ({e.reason}).")


def status():
    cfg = load()
    return {"connected": bool(cfg.get("token") and cfg.get("user_id")), "username": cfg.get("username", "")}


def connect(token, user_id="", graph_base_override=""):
    """Check the token with Instagram and remember it."""
    token = token.strip()
    cfg = {"token": token, "user_id": str(user_id).strip()}
    if graph_base_override:
        cfg["graph_base"] = graph_base_override
    base = f"{graph_base(cfg)}/{VERSION}"
    if cfg["user_id"]:
        me = _call("GET", f"{base}/{cfg['user_id']}", {"fields": "username", "access_token": token})
    else:
        me = _call("GET", f"{base}/me", {"fields": "user_id,username", "access_token": token})
        cfg["user_id"] = str(me.get("user_id") or me.get("id") or "")
    if not cfg["user_id"]:
        raise SystemExit("Instagram didn't return an account id; paste your Instagram user id too.")
    cfg["username"] = me.get("username", "")
    CONFIG.write_text(json.dumps(cfg))
    os.chmod(CONFIG, 0o600)
    return status()


def disconnect():
    CONFIG.unlink(missing_ok=True)
    return status()


def post_reel(video, caption, cover_ms=0, share_to_feed=True):
    """Publish the MP4 as a Reel. Returns {"id", "permalink"}."""
    cfg = load()
    if not (cfg.get("token") and cfg.get("user_id")):
        raise SystemExit("Connect your Instagram account first.")
    token, uid = cfg["token"], cfg["user_id"]
    base = f"{graph_base(cfg)}/{VERSION}"
    video = Path(video)
    size = video.stat().st_size
    cs.log("[instagram] preparing the upload...", 0.05)
    c = _call("POST", f"{base}/{uid}/media", {"media_type": "REELS", "upload_type": "resumable", "caption": caption,
                                              "share_to_feed": "true" if share_to_feed else "false",
                                              "thumb_offset": str(int(cover_ms)), "access_token": token})
    cid, uri = c["id"], c.get("uri") or f"https://rupload.facebook.com/ig-api-upload/{VERSION}/{c['id']}"
    if cfg.get("graph_base"):
        uri = f"{graph_base(cfg)}/ig-api-upload/{VERSION}/{cid}"
    cs.log("[instagram] uploading the video...", 0.15)
    _call("POST", uri, data=video.read_bytes(), headers={"Authorization": f"OAuth {token}", "offset": "0",
                                                          "file_size": str(size)}, timeout=600)
    cs.log("[instagram] Instagram is processing the video...", 0.5)
    for i in range(120):                              # up to ~10 minutes
        cs.checkpoint()
        st = _call("GET", f"{base}/{cid}", {"fields": "status_code,status", "access_token": token})
        code = st.get("status_code")
        if code == "FINISHED":
            break
        if code in ("ERROR", "EXPIRED"):
            raise SystemExit(f"Instagram couldn't process the video: {st.get('status', code)}")
        cs.report(min(0.85, 0.5 + i * 0.01))
        time.sleep(5)
    else:
        raise SystemExit("Instagram is still processing after 10 minutes; try publishing again later.")
    cs.log("[instagram] publishing...", 0.9)
    pub = _call("POST", f"{base}/{uid}/media_publish", {"creation_id": cid, "access_token": token})
    link = ""
    try:
        link = _call("GET", f"{base}/{pub['id']}", {"fields": "permalink", "access_token": token}).get("permalink", "")
    except SystemExit:
        pass
    cs.log("[instagram] posted", 1.0)
    return {"id": pub["id"], "permalink": link, "posted_at": time.time()}
