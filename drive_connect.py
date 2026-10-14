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
PROC = {"p": None}
_lock = threading.Lock()


def rclone():
    return shutil.which("rclone") or (str(BIN) if BIN.exists() else None)


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) DesignerStudio"})
    return urllib.request.urlopen(req, timeout=timeout).read()   # downloads.rclone.org refuses Python's default agent


def install():
    """Download the current rclone for Linux x86-64 from downloads.rclone.org, verify its SHA-256 against the
    published SHA256SUMS, and put the binary in ~/.local/bin."""
    if rclone():
        return rclone()
    base = "https://downloads.rclone.org"
    ver = _get(f"{base}/version.txt", 30).decode().split()[-1]   # e.g. v1.75.2
    name = f"rclone-{ver}-linux-amd64.zip"
    data = _get(f"{base}/{ver}/{name}", 300)
    sums = _get(f"{base}/{ver}/SHA256SUMS", 30).decode()
    want = next((ln.split()[0] for ln in sums.splitlines() if ln.strip().endswith(name)), None)
    if not want or hashlib.sha256(data).hexdigest() != want:
        raise RuntimeError("the rclone download didn't match its published checksum")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        member = next(n for n in z.namelist() if n.endswith("/rclone"))
        BIN.parent.mkdir(parents=True, exist_ok=True)
        BIN.write_bytes(z.read(member))
    os.chmod(BIN, 0o755)
    return str(BIN)


def connected(remote=REMOTE):
    exe = rclone()
    if not exe:
        return False
    out = subprocess.run([exe, "listremotes"], capture_output=True, text=True).stdout
    return f"{remote}:" in out.split()


def _run(remote):
    try:
        STATE.update(status="installing", url="", error="")
        exe = install()
        STATE["status"] = "starting"
        subprocess.run([exe, "config", "delete", remote], capture_output=True)
        p = subprocess.Popen([exe, "config", "create", remote, "drive", "scope=drive.readonly", "config_is_local=true"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        PROC["p"] = p
        deadline = time.time() + 600
        for line in p.stdout:
            m = re.search(r"(http://127\.0\.0\.1:53682/auth\?state=\S+)", line)
            if m:
                STATE.update(status="waiting", url=m.group(1))
            if time.time() > deadline:
                p.kill()
                break
        code = p.wait()
        if code == 0 and connected(remote):
            STATE.update(status="connected", url="")
        else:
            STATE.update(status="error", error="Google sign-in didn't finish. Try again.")
    except Exception as e:  # noqa: BLE001
        STATE.update(status="error", error=f"Couldn't connect Google Drive ({str(e)[:120]}).")


def start(remote=REMOTE):
    """Begin connecting in the background; poll status()."""
    with _lock:
        if STATE["status"] in ("installing", "starting", "waiting") and time.time() - STATE["started"] < 600:
            return status(remote)
        STATE["started"] = time.time()
        threading.Thread(target=_run, args=(remote,), daemon=True).start()
    time.sleep(1.5)                                  # usually enough for the sign-in link to appear
    return status(remote)


def status(remote=REMOTE):
    s = dict(STATE)
    if s["status"] not in ("installing", "starting", "waiting"):
        s["status"] = "connected" if connected(remote) else ("error" if s["status"] == "error" else "idle")
    s["remote"] = remote
    return s


def disconnect(remote=REMOTE):
    p = PROC.get("p")
    if p and p.poll() is None:                       # an unfinished sign-in
        p.kill()
    exe = rclone()
    if exe:
        subprocess.run([exe, "config", "delete", remote], capture_output=True)
    STATE.update(status="idle", url="", error="")
    return status(remote)
