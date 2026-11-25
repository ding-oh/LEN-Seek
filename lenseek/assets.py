"""Download helper for the large release assets listed in lenseek/configs/assets.yaml.

Each asset has a file name, size and SHA-256, and a URL (``null`` while it is not published). Files are stored in
``$LENSEEK_HOME/assets`` (default ``~/.cache/lenseek/assets``) and verified after download. ``kind: tar`` assets are
unpacked next to the archive into a directory named after the asset.

``$LENSEEK_ASSET_MIRROR`` overrides the URLs: a local directory (files are linked from it by file name) or a base
URL (``<mirror>/<file name>``). Useful for offline machines and for testing before publication.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, Optional

from .resources import asset_registry


class AssetError(RuntimeError):
    pass


def home() -> Path:
    return Path(os.environ.get("LENSEEK_HOME", Path.home() / ".cache" / "lenseek"))


def asset_dir() -> Path:
    d = home() / "assets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def registry() -> Dict[str, dict]:
    return {k: v for k, v in asset_registry().get("assets", {}).items()}


def _sha256(path: Path, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def _verify(path: Path, entry: dict, name: str) -> None:
    size = entry.get("size")
    if size is not None and path.stat().st_size != int(size):
        raise AssetError(f"{name}: {path} has {path.stat().st_size} bytes, expected {size}")
    sha = entry.get("sha256")
    if sha and _sha256(path) != sha:
        raise AssetError(f"{name}: SHA-256 of {path} does not match lenseek/configs/assets.yaml")


def _download(url: str, dest: Path, quiet: bool) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
            done += len(b)
            if not quiet and total:
                sys.stderr.write(f"\r  {dest.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB")
        if not quiet:
            sys.stderr.write("\n")
    tmp.rename(dest)


def fetch(name: str, quiet: bool = False, verify: bool = True) -> Path:
    """Local path of an asset (downloaded and verified if needed). Tar assets return the unpacked directory."""
    reg = registry()
    if name not in reg:
        raise AssetError(f"unknown asset {name!r}; known: {', '.join(sorted(reg))}")
    entry = reg[name]
    if not entry.get("file"):
        raise AssetError(f"{name} is not released ({entry.get('source', 'see lenseek/configs/assets.yaml')})")
    dest = asset_dir() / entry["file"]
    if not dest.exists():
        mirror = os.environ.get("LENSEEK_ASSET_MIRROR")
        if mirror and os.path.isdir(mirror):
            src = Path(mirror) / entry["file"]
            if not src.exists():
                raise AssetError(f"{name}: {src} not found in LENSEEK_ASSET_MIRROR")
            if not quiet:
                print(f"[lenseek-fetch] linking {name} from {src} into {dest.parent}", file=sys.stderr)
            try:
                os.symlink(src.resolve(), dest)
            except OSError:
                shutil.copy(src, dest)
        else:
            url = f"{mirror.rstrip('/')}/{entry['file']}" if mirror else entry.get("url")
            if not url:
                raise AssetError(f"{name} ({entry['file']}) is not published yet: its URL in "
                                 f"lenseek/configs/assets.yaml is empty. Set LENSEEK_ASSET_MIRROR to a directory or URL "
                                 f"that holds it.")
            if not quiet:
                print(f"[lenseek-fetch] downloading {name} from {url}", file=sys.stderr)
            try:
                _download(url, dest, quiet)
            except OSError as e:                    # URLError and HTTPError, e.g. a record that is not public yet
                dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
                raise AssetError(f"{name}: download from {url} failed ({e}); set LENSEEK_ASSET_MIRROR to a directory "
                                 f"or URL that holds {entry['file']}") from None
        if verify:
            try:
                _verify(dest, entry, name)
            except AssetError:
                dest.unlink()
                raise
    if entry.get("kind") == "tar":
        out = asset_dir() / name
        marker = out / ".unpacked"
        if marker.exists() and entry.get("sha256") and marker.read_text().strip() not in ("", entry["sha256"]):
            raise AssetError(f"{name}: {out} was unpacked from another version of {entry['file']}; delete {out} "
                             f"(and {dest} if it is a copy) and fetch again")
        if not marker.exists():
            out.mkdir(exist_ok=True)
            if not quiet:
                print(f"[lenseek-fetch] unpacking {entry['file']} into {out}", file=sys.stderr)
            with tarfile.open(dest) as t:
                try:
                    t.extractall(out, filter="data")
                except TypeError:  # Python without tarfile extraction filters
                    t.extractall(out)
            marker.write_text(entry.get("sha256", ""))
        inner = entry.get("root")
        return out / inner if inner else out
    return dest


def local_path(name: str) -> Optional[Path]:
    """Path of an asset if it is already present (no download)."""
    entry = registry().get(name)
    if entry is None or not entry.get("file"):
        return None
    if entry.get("kind") == "tar":
        out = asset_dir() / name
        return (out / entry["root"] if entry.get("root") else out) if (out / ".unpacked").exists() else None
    p = asset_dir() / entry["file"]
    return p if p.exists() else None
