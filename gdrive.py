"""Google Drive share links without an account: files download from Google's direct-download address and folders
are listed from Drive's plain folder view. (yt-dlp's Drive support gets "403 Forbidden" from Google, so it isn't used.)
Works for anything shared as "Anyone with the link"."""
import html
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}
VIDEO_RE = re.compile(r"\.(mp4|mov|mkv|webm|m4v|avi|mts|m2ts)$", re.I)


def is_drive(url):
    return bool(re.match(r"https?://(drive|docs)\.google\.com/", url or ""))


def parse(url):
    """("file" | "folder", id) from any Drive share link."""
    m = re.search(r"/folders/([\w-]{10,})", url)
    if m:
        return "folder", m.group(1)
    m = re.search(r"/file/d/([\w-]{10,})", url) or re.search(r"[?&]id=([\w-]{10,})", url)
    if m:
        return "file", m.group(1)
    raise ValueError("That doesn't look like a Google Drive file or folder link.")


def _open(url, headers=None, timeout=30):
    return urllib.request.urlopen(urllib.request.Request(url, headers={**UA, **(headers or {})}), timeout=timeout)


def download_url(fid):
    return f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"


def file_info(fid):
    """Name, size and type of a shared file (reads only the first bytes)."""
    try:
        with _open(download_url(fid), {"Range": "bytes=0-0"}) as r:
            ctype = r.headers.get("Content-Type", "")
            cd = r.headers.get("Content-Disposition", "")
            total = r.headers.get("Content-Range", "").split("/")[-1]
    except urllib.error.HTTPError as e:
        raise ValueError(_denied(e.code))
    if "text/html" in ctype:
        raise ValueError("Google wouldn't hand over that file. Check it's shared as \"Anyone with the link\".")
    name = ""
    m = re.search(r"filename\*=UTF-8''([^;]+)", cd) or re.search(r'filename="?([^";]+)"?', cd)
    if m:
        name = urllib.parse.unquote(m.group(1))
    return {"id": fid, "name": name or f"drive-{fid}.mp4", "size": int(total) if total.isdigit() else 0, "type": ctype}


def _denied(code):
    if code in (401, 403):
        return "That Drive file isn't shared publicly. In Drive: Share → General access → Anyone with the link."
    if code == 404:
        return "Google Drive says that file or folder doesn't exist. Check the link."
    return f"Google Drive returned an error ({code})."


def list_folder(fid):
    """Files in a shared folder: [{"id", "name", "video"}] (one level)."""
    try:
        with _open(f"https://drive.google.com/embeddedfolderview?id={fid}#list") as r:
            page = r.read().decode(errors="replace")
    except urllib.error.HTTPError as e:
        raise ValueError(_denied(e.code))
    title = re.search(r"<title>(.*?)</title>", page, re.S)
    items, seen = [], set()
    for m in re.finditer(r'<a href="https://drive\.google\.com/file/d/([\w-]+)/[^"]*"[^>]*>.*?'
                         r'<div class="flip-entry-title">(.*?)</div>', page, re.S):
        i, name = m.group(1), html.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip()
        if i not in seen:
            seen.add(i)
            items.append({"id": i, "name": name, "video": bool(VIDEO_RE.search(name))})
    if not items and "flip-entry" not in page:
        raise ValueError("Couldn't open that folder. Check it's shared as \"Anyone with the link\".")
    return {"title": html.unescape(title.group(1).strip()) if title else "Drive folder", "items": items}


def download(url, dest_dir, progress=None):
    """Download a shared Drive file into dest_dir; returns the local path."""
    kind, fid = parse(url)
    if kind != "file":
        raise ValueError("That's a folder link; pick a video from it first.")
    info = file_info(fid)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w.\- ()\[\]]+", "_", info["name"]).strip() or f"drive-{fid}.mp4"
    out = dest_dir / f"{Path(safe).stem} [{fid[:8]}]{Path(safe).suffix or '.mp4'}"
    if out.exists() and info["size"] and out.stat().st_size == info["size"]:
        return out                                     # already downloaded
    tmp = out.with_suffix(out.suffix + ".part")
    done = 0
    with _open(download_url(fid), timeout=60) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(4 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress and info["size"]:
                progress(done / info["size"])
    tmp.replace(out)
    return out
