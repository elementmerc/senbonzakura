#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fetch the pinned llama.cpp artefacts, verify them, and place what the wheel needs.

WHAT THIS DOES

Reads `src/senbonzakura/vendor/pins.json`, downloads the pinned release archives for one platform
or all of them, checks each against its recorded hash (or records it the first time), extracts only
`llama-quantize` and the shared libraries it needs, and writes them to
`src/senbonzakura/vendor/bin/<platform>/`. It also fetches the conversion package at the same tag
into `vendor/src/`: the entry point, the 85 architecture modules it imports, and `gguf-py/`, which
the entry point puts on its own path. None of that is committed either; it is 87 files, and a
158-file diff at every pin bump would buy the appearance of reviewability rather than the fact.

WHY THE HASH IS RECORDED RATHER THAN CONFIGURED

The first fetch of a pin has nothing to check against, so it records what it received: tag, size,
sha256. Every later fetch of that same pin verifies against the recorded value, and the recorded
value is in a committed file, so a change to it shows up in a diff.

That is trust-on-first-use, and its weakness is worth stating plainly: it detects a file that
changed after we first saw it, not a file that was already wrong when we first saw it. The
protections against the second are elsewhere and are not this tool's job: the cooldown means a
compromised release has had a week to be noticed, HTTPS means the bytes came from GitHub, and the
size is cross-checked against the release metadata.

What it deliberately does NOT do is accept a hash a human pasted in. A hash transcribed by hand,
or read off a page by a summarising model, is a hash nobody verified, and it is indistinguishable
from a correct one until the moment it matters.

    python tools/vendor_llama.py                 # this platform
    python tools/vendor_llama.py --all           # every platform, for a release build
    python tools/vendor_llama.py --verify-only   # check what is already here, download nothing
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from senbonzakura import vendored, vendoring

ASSET_URL = "https://github.com/{repo}/releases/download/{tag}/{file}"
SOURCE_URL = "https://github.com/{repo}/archive/refs/tags/{tag}.tar.gz"
TIMEOUT_S = 300

#: A 429 or a 5xx is the upstream asking us to wait, not a reason to abandon a release build after
#: several hundred megabytes have already been fetched. Backoff is exponential from this base.
#:
#: An authenticated fetch does NOT lift this, and it is worth recording so nobody spends an
#: afternoon rediscovering it. Release archives come from codeload, which is throttled per IP and
#: separately from the REST API: measured on 2026-08-17, the core API sat at 5000/5000 while
#: `gh api .../tarball/...` returned 429 on the same machine in the same minute. A token is the
#: right answer to an API limit and no answer at all to this one. Waiting is the answer to this one.
RETRY_STATUSES = (429, 500, 502, 503, 504)
RETRIES = 5
BACKOFF_BASE_S = 4

#: The only executable we need out of an eighty-megabyte archive. Everything else in there is
#: inference and tooling this package does not use, and shipping it would be dead weight in
#: every wheel.
WANT_BIN = "llama-quantize"

#: Shared libraries to keep. Matched by prefix because the exact set changes between releases and
#: a hardcoded list silently ships a binary that cannot start.
LIB_PREFIXES = ("libggml", "libllama", "libmtmd", "ggml", "llama")

#: A shared library name at ANY version. The first attempt matched only a bare `.so` and therefore
#: extracted none of `libggml.so.0.19.0`, `libllama.so.0.0.10355` or `libllama-common.so.0.0.10355`,
#: which are the ones that actually matter. The binary then failed to start on a missing
#: `libllama-common.so.0`, and no test could have caught it because no test ran the binary.
_SONAME = re.compile(r"\.(so|dylib|dll)(\.\d+)*$")

#: Per-tool implementation libraries for tools this package does not ship. `llama-quantize` needs
#: `libllama-quantize-impl.so`; the server, cli, bench, perplexity and multimodal ones are weight
#: in every wheel for a binary nobody here invokes. Trimming is only safe because the smoke check
#: below runs the binary afterwards, so getting this wrong fails at vendor time rather than at use.
DROP_IMPL = ("libllama-batched-bench-impl", "libllama-bench-impl", "libllama-cli-impl",
             "libllama-completion-impl", "libllama-fit-params-impl", "libllama-perplexity-impl",
             "libllama-server-impl", "libmtmd")


class VendorFetchError(Exception):
    """A download or extraction failed, phrased for a person."""


def _download(url, dest, *, timeout=TIMEOUT_S, opener=None, retries=RETRIES, sleep=time.sleep):
    """Stream a URL to a path, returning (bytes, sha256). Never leaves a partial file behind.

    Retries a 429 or a 5xx with exponential backoff. A release build fetches six platform archives
    and a source tarball, so being rate-limited partway through is an ordinary event rather than an
    exceptional one, and abandoning the run means starting the whole fetch again. A 404 is not
    retried: it means the pin names something that does not exist, and waiting will not fix it.
    """
    tmp = Path(f"{dest}.part")
    last = None
    for attempt in range(retries):
        h = hashlib.sha256()
        n = 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "senbonzakura-vendor"})
            with (opener or urllib.request.urlopen)(req, timeout=timeout) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    h.update(chunk)
                    f.write(chunk)
                    n += len(chunk)
        except urllib.error.HTTPError as e:
            tmp.unlink(missing_ok=True)
            last = e
            if e.code not in RETRY_STATUSES or attempt == retries - 1:
                break
            wait = BACKOFF_BASE_S * (2 ** attempt)
            print(f"  HTTP {e.code} from {url.rsplit('/', 1)[-1]}; retrying in {wait}s "
                  f"({attempt + 1}/{retries - 1})")
            sleep(wait)
            continue
        except (urllib.error.URLError, OSError) as e:
            tmp.unlink(missing_ok=True)
            last = e
            break
        except BaseException:
            # A Ctrl+C mid-download must not leave a truncated file the next run treats as done.
            tmp.unlink(missing_ok=True)
            raise
        else:
            tmp.replace(dest)
            return n, h.hexdigest()
    raise VendorFetchError(f"could not download {url}: {last}")


def _check_or_record(pin, key, got_size, got_hash, *, expect_size=None, fatal=True, log=print):
    """Verify against the recorded hash, or record it on a first fetch. Raises on a mismatch.

    `fatal=False` is for an archive the forge GENERATES rather than stores: a source tarball at a
    tag is rebuilt on request and its bytes depend on the compression in use that day, so a change
    there is a statement about packaging and not about contents. Those callers warn here and gate
    on `content_digest` after extraction instead, which is the check that actually answers "are
    these the same files".
    """
    recorded = (pin.get("sha256") or {}).get(key)
    if expect_size is not None and got_size != expect_size:
        raise VendorFetchError(
            f"{key}: downloaded {got_size:,} bytes but the manifest records {expect_size:,}. A "
            f"release asset does not change size, so either the pin is wrong or the download is "
            f"truncated. Nothing has been extracted.")
    if recorded is None:
        log(f"  {key}: recording sha256 {got_hash} (first fetch of this pin)")
        return got_hash, True
    if recorded != got_hash:
        if not fatal:
            log(f"  {key}: the archive hashes to {got_hash} and the manifest records {recorded}. "
                f"This URL is generated on request rather than stored, so a repackaging changes "
                f"these bytes without changing a single file. Continuing to the content check, "
                f"which is the one that can tell those apart.")
            return got_hash, True
        raise VendorFetchError(
            f"{key}: sha256 is {got_hash} but the manifest records {recorded}. The file at this "
            f"URL has changed since it was pinned, which for an immutable release asset should "
            f"not happen. Do NOT extract it. Investigate before touching the pin.")
    log(f"  {key}: sha256 matches the recorded value")
    return recorded, False


@contextlib.contextmanager
def _open_archive(archive):
    """Yield (handle, [(name, member), ...]) for a tar.gz or a zip, closing it on every path.

    A context manager rather than returning the handle for the caller to close in a `finally`.
    The version that returned it was correct and unreadably so: the guarantee lived in one
    function and the handle in another, which is the shape a leak takes when either is edited.
    """
    if archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix == ".tgz":
        with tarfile.open(archive, "r:gz") as tf:
            # Symlinks are kept, not filtered. `isfile()` alone dropped every soname alias, and
            # those aliases ARE the names the binary asks the loader for: it wants
            # `libllama-common.so.0`, which exists only as a link to `...so.0.0.10355`.
            yield tf, [(m.name, m) for m in tf.getmembers() if m.isfile() or m.issym()]
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            yield zf, [(n, n) for n in zf.namelist() if not n.endswith("/")]
    else:
        raise VendorFetchError(f"{archive.name}: not a tar.gz or a zip, so it cannot be unpacked")


def _wanted(name):
    base = Path(name).name
    if base in (WANT_BIN, f"{WANT_BIN}.exe"):
        return True
    if not base.startswith(LIB_PREFIXES) or not _SONAME.search(base):
        return False
    return not base.startswith(DROP_IMPL)


def extract(archive, out_dir, *, log=print):
    """Pull llama-quantize and its libraries out, flat. Returns the names written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written, links = [], []
    with _open_archive(archive) as (handle, members):
        for name, member in members:
            if not _wanted(name):
                continue
            # Flattened to the basename on purpose. The archive nests under build/bin/, and
            # honouring a path out of an archive is how a crafted member writes outside the
            # target directory. Taking only the name makes traversal structurally impossible.
            target = out_dir / Path(name).name
            if isinstance(handle, tarfile.TarFile) and member.issym():
                # Recreate the alias rather than duplicating two megabytes of library under a
                # second name. Basename only, so a link cannot point outside the directory.
                links.append((target, Path(member.linkname).name))
                continue
            src = (handle.extractfile(member) if isinstance(handle, tarfile.TarFile)
                   else handle.open(member))
            if src is None:
                continue
            with src, open(target, "wb") as f:
                shutil.copyfileobj(src, f)
            target.chmod(0o755)
            written.append(target.name)
    # After the real files, so a link never precedes its target. A copy where symlinks are not
    # available (Windows without privilege) costs disk and works identically.
    for link, target_name in links:
        link.unlink(missing_ok=True)
        try:
            link.symlink_to(target_name)
        except (OSError, NotImplementedError):
            real = link.parent / target_name
            if real.is_file():
                shutil.copyfile(real, link)
                link.chmod(0o755)
        written.append(link.name)

    if not any(w.startswith(WANT_BIN) for w in written):
        raise VendorFetchError(
            f"{archive.name} does not contain {WANT_BIN}. The upstream archive layout has changed, "
            f"so this tool is looking in the wrong place rather than the release being broken.")
    log(f"  extracted {len(written)} file(s): {', '.join(sorted(written))}")
    return written


def smoke(out_dir, key, *, log=print):
    """Run the extracted binary. An extraction that cannot start is not a vendored binary.

    This is the check that would have caught the first attempt at this tool: it extracted
    twenty-four files, reported success, and produced a `llama-quantize` that died on a missing
    `libllama-common.so.0`, because the soname pattern matched only a bare `.so` and the version
    aliases are symlinks that an `isfile()` filter had already dropped. Nothing was wrong with the
    download and nothing was wrong with the hash. The only way to find it was to run the thing.

    Only for the CURRENT platform, since a macOS binary cannot be executed on Linux. Other
    platforms get the structural check above and their real smoke test on the machine that uses
    them, which is honest about what has and has not been verified.
    """
    exe = out_dir / (f"{WANT_BIN}.exe" if key.startswith("windows") else WANT_BIN)
    if key != vendored.platform_key():
        log(f"  {key}: not this platform, so the binary is not run here. Extraction checked only.")
        return False
    try:
        r = subprocess.run([str(exe), "--help"], capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as e:
        raise VendorFetchError(
            f"{key}: the extracted {WANT_BIN} could not be started ({e}). The archive downloaded "
            f"and hashed correctly, so this is a missing shared library rather than a bad "
            f"download: check that every soname alias was kept.") from e
    out = (r.stdout + r.stderr).decode("utf-8", errors="replace")
    if "usage" not in out.lower():
        raise VendorFetchError(
            f"{key}: {WANT_BIN} started but did not print a usage message; it exited "
            f"{r.returncode} saying {out.strip()[:200]!r}. A binary that cannot describe itself "
            f"will not quantise anything either.")
    log(f"  {key}: {WANT_BIN} runs and reports its usage")
    return True


def vendor_binaries(manifest, keys, *, dry_run=False, verify_only=False, log=print):
    """Fetch, verify and extract the binary pin for each platform key. Returns hashes to record."""
    pin = manifest["pins"]["llama.cpp"]
    recorded = dict(pin.get("sha256") or {})
    with tempfile.TemporaryDirectory(prefix="senbon-vendor-") as td:
        for key in keys:
            asset = (pin["assets"] or {}).get(key)
            if asset is None:
                log(f"  {key}: no asset declared for this platform, skipping")
                continue
            url = ASSET_URL.format(repo=pin["repo"], tag=pin["tag"], file=asset["file"])
            log(f"{key}: {asset['file']}")
            if dry_run:
                log(f"  would fetch {url}")
                continue
            local = Path(td) / asset["file"]
            size, digest = _download(url, local)
            recorded[key], _ = _check_or_record(pin, key, size, digest,
                                                expect_size=asset.get("size"), log=log)
            if verify_only:
                continue
            out_dir = vendored.VENDOR_BIN / key
            extract(local, out_dir, log=log)
            smoke(out_dir, key, log=log)
    return recorded


def vendor_conversion(manifest, *, dry_run=False, verify_only=False, log=print):
    """Fetch the conversion package from the source archive at the pinned tag.

    One download rather than 158 API calls for 158 files. The archive is hashed as a whole, so the
    pin covers every file in the package at once and a single changed byte anywhere in it shows up.

    Why the whole package and not just the script: as of b10355 `convert_hf_to_gguf.py` is a
    307-line entry point whose first real statement is `from conversion import ...`. Vendoring only
    the named file produces a stub that raises ImportError, which is what the first version of this
    tool did, silently and with a green exit code.
    """
    pin = manifest["pins"]["llama_conversion"]
    url = SOURCE_URL.format(repo=pin["repo"], tag=pin["tag"])
    keep = pin["assets"]["any"]["keep"]
    log(f"conversion package: {pin['repo']} @ {pin['tag']} ({', '.join(keep)})")
    if dry_run:
        log(f"  would fetch {url}")
        return dict(pin.get("sha256") or {})

    with tempfile.TemporaryDirectory(prefix="senbon-vendor-src-") as td:
        archive = Path(td) / "src.tar.gz"
        size, digest = _download(url, archive)
        recorded = dict(pin.get("sha256") or {})
        recorded["any"], _ = _check_or_record(pin, "any", size, digest, fatal=False, log=log)
        if verify_only:
            return recorded

        # Wipe first: a stale module left behind from an older tag would be importable and would
        # shadow nothing, which is worse than absent because it looks current.
        dest = vendored.VENDOR_SRC
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)

        wanted = tuple(k.rstrip("/") for k in keep)
        written = 0
        with tarfile.open(archive, "r:gz") as tf:
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                # The archive nests everything under `<repo>-<tag>/`; strip that one leading
                # component and match against the paths the pin names.
                rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
                if not (rel in wanted or any(rel.startswith(w + "/") for w in wanted)):
                    continue
                # Resolved and checked, so a crafted member cannot write outside the directory.
                target = (dest / rel).resolve()
                if not str(target).startswith(str(dest.resolve()) + "/"):
                    raise VendorFetchError(
                        f"the archive contains a member that resolves outside the destination "
                        f"({m.name}); refusing to extract it")
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                if src is None:
                    continue
                with src, open(target, "wb") as f:
                    shutil.copyfileobj(src, f)
                written += 1
        log(f"  wrote {written} file(s) to {dest.relative_to(ROOT)}")
        if written < 2:
            raise VendorFetchError(
                f"only {written} file(s) matched {keep} in the source archive. The upstream layout "
                f"has moved, and a partial conversion package imports and then fails at use.")
        try:
            recorded["content"], _ = _check_or_record_content(pin, dest, log=log)
            _smoke_conversion(dest, log=log)
        except VendorFetchError:
            # A package that failed its own verification must not be left on disk. It would be
            # importable, it would look current, and the next run would use it without re-checking
            # anything. Absent is recoverable; present-and-unverified is not.
            shutil.rmtree(dest, ignore_errors=True)
            raise
    return recorded


def content_digest(root):
    """A hash over the EXTRACTED files, independent of how they were packaged.

    The archive hash covers the wrapper. This covers the contents, and the difference matters
    because the two URLs this tool fetches are not the same kind of object. A release asset is
    uploaded once and is genuinely immutable, so hashing the bytes is the right check. A source
    archive at `/archive/refs/tags/` is GENERATED on request, and its exact bytes depend on the
    compression the forge happens to use that day; a pin on those bytes can fail on a package whose
    contents never changed, and a check that cries wolf is a check people learn to override.

    Path and size go into the hash alongside the bytes, so a renamed or truncated file cannot be
    concealed by another one's contents. Sorted, so the walk order of the filesystem cannot change
    the answer between two machines.
    """
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        data = p.read_bytes()
        h.update(f"{rel}\0{len(data)}\0".encode())
        h.update(data)
    return h.hexdigest()


def _check_or_record_content(pin, dest, *, log=print):
    """Verify the extracted package against its recorded content hash, or record it on a first run.

    Kept separate from `_check_or_record` because the two failures mean different things and want
    different words. A wrapper mismatch with a matching content hash is a re-packaged archive and
    is harmless. A content mismatch is the one that matters.
    """
    got = content_digest(dest)
    recorded = (pin.get("sha256") or {}).get("content")
    if recorded is None:
        log(f"  content: recording sha256 {got} (first extraction of this pin)")
        return got, True
    if recorded != got:
        raise VendorFetchError(
            f"the extracted package hashes to {got} and the manifest records {recorded}. The "
            f"FILES differ, not just the packaging, so this is not a re-compressed archive: at a "
            f"pinned tag the contents cannot legitimately change. Investigate before touching the "
            f"pin; nothing downstream should use this package.")
    log("  content: sha256 matches the recorded value")
    return recorded, False


def _smoke_conversion(dest, log=print):
    """Import the vendored entry point. A package that cannot import is not vendored.

    The check the first attempt needed: it wrote a 307-line stub, reported success, and would have
    failed at `from conversion import ...` the first time anyone converted a model.
    """
    script = dest / "convert_hf_to_gguf.py"
    if not script.is_file():
        raise VendorFetchError(f"{script.name} is not in the vendored package")
    r = subprocess.run([sys.executable, str(script), "--help"],
                       capture_output=True, timeout=180, check=False, cwd=str(dest))
    out = (r.stdout + r.stderr).decode("utf-8", errors="replace")
    if "usage" not in out.lower():
        raise VendorFetchError(
            f"the vendored converter could not run: exit {r.returncode}, saying "
            f"{out.strip()[-400:]!r}. If this is a missing `gguf` module, that is a declared "
            f"dependency and the environment needs it; if it is an ImportError from `conversion`, "
            f"the package was extracted incompletely.")
    log("  conversion package imports and reports its usage")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true",
                    help="every declared platform, which is what a release build needs")
    ap.add_argument("--platform", default=None, help="one platform key, e.g. linux-x86_64")
    ap.add_argument("--verify-only", action="store_true",
                    help="download and check hashes, but extract nothing")
    ap.add_argument("--dry-run", action="store_true", help="print the URLs and stop")
    ap.add_argument("--skip-script", action="store_true", help="binaries only")
    ap.add_argument("--manifest", default=None)
    a = ap.parse_args(argv)

    try:
        manifest = vendoring.load_manifest(a.manifest)
    except vendoring.VendorError as e:
        print(f"vendor: {e}", file=sys.stderr)
        return 1

    pin = manifest["pins"]["llama.cpp"]
    if a.all:
        keys = sorted(pin["assets"])
    elif a.platform:
        keys = [a.platform]
    else:
        k = vendored.platform_key()
        if k is None:
            print("vendor: this platform is not one the pins cover, and nothing was built for it. "
                  "Pass --platform to fetch a specific one anyway.", file=sys.stderr)
            return 1
        keys = [k]

    print(f"llama.cpp pinned at {pin['tag']} ({pin['published'][:10]}), fetching: {', '.join(keys)}")
    try:
        hashes = vendor_binaries(manifest, keys, dry_run=a.dry_run,
                                 verify_only=a.verify_only)
        script_hashes = ({} if a.skip_script else
                         vendor_conversion(manifest, dry_run=a.dry_run, verify_only=a.verify_only))
    except VendorFetchError as e:
        print(f"vendor: {e}", file=sys.stderr)
        return 1

    if a.dry_run:
        return 0

    # Record what was actually received, into the committed manifest, so the next run verifies.
    path = Path(a.manifest or vendoring.MANIFEST)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["pins"]["llama.cpp"]["sha256"] = dict(sorted(hashes.items()))
    if script_hashes:
        doc["pins"]["llama_conversion"]["sha256"] = dict(sorted(script_hashes.items()))
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"recorded {len(hashes)} binary hash(es) in {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
