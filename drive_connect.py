"""One-click "Connect Google Drive": installs rclone if needed (official download, checksum verified), then runs
rclone's Google sign-in (read-only access) and creates the "gdrive" remote the import screens browse.
No Google Cloud project or typing commands needed: rclone's own sign-in is used."""
import hashlib
import io
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

BIN = Path.home() / ".local" / "bin" / "rclone"
REMOTE = "gdrive"
STATE = {"status": "idle", "url": "", "error": "", "started": 0}
_lock = threading.Lock()


def rclone():
    return shutil.which("rclone") or (str(BIN) if BIN.exists() else None)


def install():
    """Download the current rclone for Linux x86-64 from downloads.rclone.org, verify its SHA-256 against the
    published SHA256SUMS, and put the binary in ~/.local/bin."""
    if rclone():
        return rclone()
    base = "https://downloads.rclone.org"
    ver = urllib.request.urlopen(f"{base}/version.txt", timeout=30).read().decode().split()[-1]   # e.g. v1.71.1
    name = f"rclone-{ver}-linux-amd64.zip"
    data = urllib.request.urlopen(f"{base}/{ver}/{name}", timeout=300).read()
    sums = urllib.request.urlopen(f"{base}/{ver}/SHA256SUMS", timeout=30).read().decode()
    want = next((ln.split()[0] for ln in sums.splitlines() if ln.strip().endswith(name)), None)
    if not want or hashlib.sha256(data).hexdigest() != want:
        raise RuntimeError("the rclone download didn't match its published checksum")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        member = next(n for n in z.namelist() if n.endswith("/rclone"))
        BIN.parent.mkdir(parents=True, exist_ok=True)
        BIN.write_bytes(z.read(member))
    os.chmod(BIN, 0o755)
    return str(BIN)


def connected():
    exe = rclone()
    if not exe:
        return False
    out = subprocess.run([exe, "listremotes"], capture_output=True, text=True).stdout
    return f"{REMOTE}:" in out.split()


def _run():
    try:
        STATE.update(status="installing", url="", error="")
        exe = install()
        STATE["status"] = "starting"
        subprocess.run([exe, "config", "delete", REMOTE], capture_output=True)
        p = subprocess.Popen([exe, "config", "create", REMOTE, "drive", "scope=drive.readonly", "config_is_local=true"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 600
        for line in p.stdout:
            m = re.search(r"(http://127\.0\.0\.1:53682/auth\?state=\S+)", line)
            if m:
                STATE.update(status="waiting", url=m.group(1))
            if time.time() > deadline:
                p.kill()
                break
        code = p.wait()
        if code == 0 and connected():
            STATE.update(status="connected", url="")
        else:
            STATE.update(status="error", error="Google sign-in didn't finish. Try again.")
    except Exception as e:  # noqa: BLE001
        STATE.update(status="error", error=f"Couldn't connect Google Drive ({str(e)[:120]}).")


def start():
    """Begin connecting in the background; poll status()."""
    with _lock:
        if STATE["status"] in ("installing", "starting", "waiting") and time.time() - STATE["started"] < 600:
            return status()
        STATE["started"] = time.time()
        threading.Thread(target=_run, daemon=True).start()
    time.sleep(1.5)                                  # usually enough for the sign-in link to appear
    return status()


def status():
    s = dict(STATE)
    if s["status"] not in ("installing", "starting", "waiting"):
        s["status"] = "connected" if connected() else ("error" if s["status"] == "error" else "idle")
    s["remote"] = REMOTE
    return s


def disconnect():
    exe = rclone()
    if exe:
        subprocess.run([exe, "config", "delete", REMOTE], capture_output=True)
    STATE.update(status="idle", url="", error="")
    return status()
