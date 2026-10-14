"""Designer Studio users: profiles (each with their own preferences, creator profile, Instagram and Drive
connections and projects), optional passcodes, sign-in sessions and an activity log.

Data lives in users/ next to this file (never committed):
  users/users.json            [{id, name, admin, created, passcode (scrypt) or null, photo}]
  users/sessions.json         {token: {user, created, seen}}
  users/<id>/prefs.json       defaults chosen in the new-user setup (styles, socials, import...)
  users/<id>/profile.json     creator profile the AI writes for (niche, audience, tone...)
  users/<id>/instagram.json   Instagram connection
  logs/activity.jsonl         one line per action: time, user, action, detail
"""
import hashlib
import json
import os
import re
import secrets
import shutil
import threading
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIR = Path(os.environ.get("DS_USERS_DIR") or HERE / "users")
INDEX = DIR / "users.json"
SESSIONS = DIR / "sessions.json"
LOG = Path(os.environ.get("DS_ACTIVITY_LOG") or HERE / "logs" / "activity.jsonl")
SESSION_DAYS = 60
_lock = threading.Lock()


def _read(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def all_users():
    return _read(INDEX, [])


def public(u):
    return {"id": u["id"], "name": u["name"], "admin": u.get("admin", False), "has_passcode": bool(u.get("passcode")),
            "photo": bool(u.get("photo")), "created": u.get("created")}


def get(uid):
    return next((u for u in all_users() if u["id"] == uid), None)


def user_dir(uid):
    d = DIR / uid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hash(code, salt=None):
    salt = salt or secrets.token_hex(16)
    h = hashlib.scrypt(code.encode(), salt=bytes.fromhex(salt), n=2 ** 14, r=8, p=1).hex()
    return f"scrypt${salt}${h}"


def check_passcode(u, code):
    if not u.get("passcode"):
        return True
    _, salt, h = u["passcode"].split("$")
    return secrets.compare_digest(_hash(code or "", salt).split("$")[2], h)


def create(name, passcode=""):
    name = re.sub(r"\s+", " ", (name or "").strip())[:40]
    if not name:
        raise ValueError("Enter a name.")
    if passcode and not re.fullmatch(r"\d{4,8}", passcode):
        raise ValueError("A passcode is 4 to 8 digits.")
    with _lock:
        users = all_users()
        if any(u["name"].lower() == name.lower() for u in users):
            raise ValueError("Someone already uses that name.")
        u = {"id": uuid.uuid4().hex[:10], "name": name, "admin": not users, "created": time.time(),
             "passcode": _hash(passcode) if passcode else None, "photo": False}
        users.append(u)
        _write(INDEX, users)
    if u["admin"]:
        _adopt_existing(u["id"])
    return u


def _adopt_existing(uid):
    """The first user inherits what was set up before users existed (creator profile, Instagram)."""
    d = user_dir(uid)
    for name in ("profile.json", "instagram.json"):
        old = HERE / name
        if old.exists() and not (d / name).exists():
            shutil.copy(old, d / name)


def update(uid, **fields):
    with _lock:
        users = all_users()
        u = next(x for x in users if x["id"] == uid)
        if "name" in fields:
            n = re.sub(r"\s+", " ", (fields["name"] or "").strip())[:40]
            if not n:
                raise ValueError("Enter a name.")
            if any(x["name"].lower() == n.lower() and x["id"] != uid for x in users):
                raise ValueError("Someone already uses that name.")
            u["name"] = n
        if "passcode" in fields:
            code = fields["passcode"] or ""
            if code and not re.fullmatch(r"\d{4,8}", code):
                raise ValueError("A passcode is 4 to 8 digits.")
            u["passcode"] = _hash(code) if code else None
        if "photo" in fields:
            u["photo"] = bool(fields["photo"])
        _write(INDEX, users)
        return u


def delete(uid):
    with _lock:
        users = [u for u in all_users() if u["id"] != uid]
        _write(INDEX, users)
        sessions = {t: s for t, s in _read(SESSIONS, {}).items() if s["user"] != uid}
        _write(SESSIONS, sessions)
    shutil.rmtree(DIR / uid, ignore_errors=True)


# ---------------------------------------------------------------- sessions

def new_session(uid):
    token = secrets.token_urlsafe(32)
    with _lock:
        s = _read(SESSIONS, {})
        now = time.time()
        s = {t: v for t, v in s.items() if now - v.get("seen", 0) < SESSION_DAYS * 86400}
        s[token] = {"user": uid, "created": now, "seen": now}
        _write(SESSIONS, s)
    return token


def session_user(token):
    if not token:
        return None
    s = _read(SESSIONS, {}).get(token)
    if not s or time.time() - s.get("seen", 0) > SESSION_DAYS * 86400:
        return None
    return get(s["user"])


def end_session(token):
    with _lock:
        s = _read(SESSIONS, {})
        s.pop(token, None)
        _write(SESSIONS, s)


# ---------------------------------------------------------------- preferences

DEFAULT_PREFS = {
    "settings": {},                              # Settings fields: captions, look, aspect, count, min, max, logo...
    "socials": {"platforms": ["instagram"], "handles": {}, "primary": "instagram", "per_week": 3,
                "times": ["evening"]},
    "import": {"default_source": "lib", "folders": []},
    "images": {"broll_style": "cinematic", "presenter": ""},
}


def prefs(uid):
    p = _read(user_dir(uid) / "prefs.json", {})
    out = json.loads(json.dumps(DEFAULT_PREFS))
    for k, v in p.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def save_prefs(uid, new):
    cur = prefs(uid)
    for k, v in (new or {}).items():
        if k in DEFAULT_PREFS and isinstance(v, dict):
            cur[k].update(v)
    _write(user_dir(uid) / "prefs.json", cur)
    return cur


# ---------------------------------------------------------------- activity log

def log(user, action, detail=""):
    """Append one line to the activity log (user may be a dict, an id or None for the system)."""
    if isinstance(user, str):
        user = get(user) or {"id": user, "name": "?"}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"t": round(time.time(), 1), "user": (user or {}).get("id"), "name": (user or {}).get("name", "system"),
           "action": action, "detail": str(detail)[:300]}
    with _lock, open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def activity(uid=None, limit=300):
    try:
        lines = LOG.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for ln in reversed(lines):
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if uid is None or r.get("user") == uid:
            out.append(r)
            if len(out) >= limit:
                break
    return out
