#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Download a model and prove the bytes are the ones the Hub published.

WHY THIS EXISTS

On 2026-08-15 a sister project lost a day to a checkpoint whose file was the exact right length
and the wrong content: the container header parsed, all 266 tensors loaded, and the model emitted
one repeated token forever. The published SHA-256 was 36587fdf...; the file on disk hashed to
aed81383... at a byte length of 2874779456, which matched the published length exactly. The
control that proved it was the file rather than the runtime was a second quantisation of the same
model, which matched its own hash and worked on the same binary throughout.

Size proves nothing. A parsing header proves nothing. Nothing downstream of a corrupt checkpoint
can tell you the checkpoint is the problem: a model that produces garbage after abliteration looks
exactly like an abliteration that went wrong, and this project would have spent the GPU time
before finding out.

So the hash is checked BEFORE the first forward pass, against the value the Hub publishes for each
LFS file at `/api/models/<repo>?blobs=true` under `siblings[].lfs.sha256`.

Usage:
    python tools/fetch_model.py LiquidAI/LFM2.5-350M --out ~/models/LFM2.5-350M
    python tools/fetch_model.py <repo> --out <dir> --verify-only
"""
import argparse
import contextlib
import errno
import hashlib
import json
import os
import sys
import urllib.request

API = "https://huggingface.co/api/models/{repo}?blobs=true"
CHUNK = 1 << 22


def published_hashes(repo, token=None):
    """{filename: sha256} for every LFS file the Hub lists, which is every weight file.

    Small files (config.json, tokenizer configs) are not LFS and carry no published sha256, so
    they are absent here by design rather than by oversight; the weights are what this guards.
    """
    url = API.format(repo=repo)
    req = urllib.request.Request(url, headers={"User-Agent": "senbonzakura"})  # noqa: S310 - fixed https host
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r:      # noqa: S310 - fixed https host
        doc = json.load(r)
    out = {}
    for sib in doc.get("siblings") or []:
        lfs = sib.get("lfs") or {}
        digest = lfs.get("sha256") or lfs.get("oid")
        if digest:
            out[sib["rfilename"]] = digest
    if not out:
        raise SystemExit(
            f"fetch: {repo} lists no LFS files with a published sha256, so nothing here could be "
            f"verified. Refusing rather than downloading unverifiable weights.")
    return out


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify(directory, expected, log=print):
    """Hash every file we have a published digest for. Returns the list of failures."""
    bad, checked = [], 0
    for name, digest in sorted(expected.items()):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            bad.append((name, "missing", digest))
            continue
        got = sha256_of(path)
        checked += 1
        if got != digest:
            bad.append((name, got, digest))
        else:
            log(f"  ok    {name}  {got[:12]}...")
    log(f"  verified {checked} of {len(expected)} published file(s)")
    return bad


@contextlib.contextmanager
def exclusive(directory, log=print):
    """Hold a lock for this destination, so two fetchers cannot race into one path.

    A sister project lost a day to a 987 MB fragment of a 5.16 GB GGUF that passed an
    "exists and is non-empty" check, and the cause was two copies of the fetcher writing the
    same file. Hashing catches a bad result AFTER the download; this stops the second writer
    starting at all, which is cheaper and removes the case where both finish and the hash is of
    a file neither of them wrote alone.

    The lock is advisory and process-scoped: it protects against this tool racing itself, which
    is the failure that actually happened, not against an unrelated program writing the same
    directory.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, ".senbon-fetch.lock")
    try:
        import fcntl
    except ImportError:              # Windows: no fcntl, so no lock and an honest warning
        log("fetch: NOTE file locking is unavailable on this platform, so two fetchers writing "
            "the same directory would not be stopped. Run one at a time.")
        yield
        return
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            if e.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            raise SystemExit(
                f"fetch: another fetch is already writing {directory} (lock held on {path}). "
                f"Two fetchers racing into one path is how a truncated checkpoint gets written, "
                f"so this one stops rather than joining in. Wait for it, or choose another "
                f"--out.") from e
        os.write(fd, f"{os.getpid()}\n".encode())
        yield
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="fetch_model",
        description="Download a model and verify every weight file against its published SHA-256.")
    ap.add_argument("repo", help="Hub repo id, e.g. LiquidAI/LFM2.5-350M")
    ap.add_argument("--out", required=True, help="where the checkpoint goes")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip the download and hash what is already in --out")
    ap.add_argument("--token-env", default="HF_TOKEN",
                    help="environment variable holding a Hub token, for gated repos")
    a = ap.parse_args(argv)

    token = os.environ.get(a.token_env) or None
    print(f"fetch: reading published hashes for {a.repo}")
    expected = published_hashes(a.repo, token)
    print(f"fetch: the Hub publishes a sha256 for {len(expected)} file(s)")

    if not a.verify_only:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise SystemExit(
                "fetch: huggingface_hub is not installed, so there is nothing to download with. "
                "Install it, or download separately and re-run with --verify-only.") from None
        with exclusive(a.out):
            print(f"fetch: downloading into {a.out}")
            snapshot_download(repo_id=a.repo, local_dir=a.out, token=token)

    print("fetch: verifying")
    bad = verify(a.out, expected)
    if bad:
        for name, got, want in bad:
            print(f"  FAIL  {name}\n        got  {got}\n        want {want}", file=sys.stderr)
        raise SystemExit(
            f"fetch: {len(bad)} file(s) do not match what the Hub published. Do NOT spend compute "
            f"on this checkpoint. A file of the right length with the wrong content parses, loads "
            f"and then produces garbage that looks exactly like a failed abliteration.")
    print(f"fetch: OK, {a.repo} matches its published hashes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
