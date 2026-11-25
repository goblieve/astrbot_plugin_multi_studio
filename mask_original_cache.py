"""Short-lived cache for original images uploaded through the mask editor."""

import hashlib
import json
import re
import stat
import tempfile
import threading
import time
from pathlib import Path

from .masked_workflow import MAX_MASK_SOURCE_BYTES, MaskOriginal


CACHE_TTL_SECONDS = 24 * 60 * 60
CACHE_MAX_BYTES = 512 * 1024 * 1024
ENTRY_MAX_BYTES = MAX_MASK_SOURCE_BYTES + 16 * 1024 * 1024
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class MaskOriginalCache:
    def __init__(self, directory: Path):
        self.directory = directory
        self.lock = threading.Lock()

    def _path(self, owner, digest):
        origin, user = owner
        bot = origin.partition(":")[0]
        key = hashlib.sha256(json.dumps([bot, user, digest], ensure_ascii=False).encode("utf-8")).hexdigest()
        return self.directory / (key + ".cache")

    def get(self, owner, digest):
        path = self._path(owner, digest)
        with self.lock:
            try:
                info = path.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_size > ENTRY_MAX_BYTES:
                    return None
                if info.st_mtime < time.time() - CACHE_TTL_SECONDS:
                    path.unlink(missing_ok=True)
                    return None
                with path.open("rb") as file:
                    metadata = json.loads(file.readline(2048))
                    preview_size = metadata["preview_bytes"]
                    if type(preview_size) is not int or not 0 < preview_size <= 16 * 1024 * 1024:
                        return None
                    preview = file.read(preview_size)
                    data = file.read(MAX_MASK_SOURCE_BYTES + 1)
                if (len(preview) != preview_size or len(data) > MAX_MASK_SOURCE_BYTES
                        or hashlib.sha256(data).hexdigest() != digest):
                    return None
                return MaskOriginal(
                    data, tuple(metadata["size"]), metadata["extension"], preview,
                    tuple(metadata["preview_size"]), metadata["preview_extension"],
                )
            except (OSError, ValueError, KeyError, TypeError):
                return None

    def put(self, owner, original):
        if len(original.data) + len(original.preview) + 2048 > ENTRY_MAX_BYTES:
            return
        digest = hashlib.sha256(original.data).hexdigest()
        path = self._path(owner, digest)
        metadata = {
            "size": original.size, "extension": original.extension,
            "preview_size": original.preview_size, "preview_extension": original.preview_extension,
            "preview_bytes": len(original.preview),
        }
        with self.lock:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            staged = None
            try:
                with tempfile.NamedTemporaryFile(prefix=".original-", dir=self.directory, delete=False) as file:
                    staged = Path(file.name)
                    file.write(json.dumps(metadata).encode("ascii") + b"\n")
                    file.write(original.preview)
                    file.write(original.data)
                staged.replace(path)
            finally:
                if staged is not None:
                    staged.unlink(missing_ok=True)
            self._prune()

    def prune(self):
        with self.lock:
            self._prune()

    def _prune(self):
        if not self.directory.is_dir():
            return
        cutoff = time.time() - CACHE_TTL_SECONDS
        entries = []
        for path in self.directory.iterdir():
            is_entry = path.suffix == ".cache" and SHA256_PATTERN.fullmatch(path.stem)
            if not (is_entry or path.name.startswith(".original-")):
                continue
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            if info.st_mtime < cutoff:
                path.unlink(missing_ok=True)
            elif is_entry:
                entries.append((info.st_mtime, info.st_size, path))
        used = sum(size for _, size, _ in entries)
        for _, size, path in sorted(entries):
            if used <= CACHE_MAX_BYTES:
                break
            path.unlink(missing_ok=True)
            used -= size
