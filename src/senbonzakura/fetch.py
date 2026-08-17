# SPDX-License-Identifier: AGPL-3.0-or-later
"""`senbonzakura fetch`: get a model file and prove it is the file you asked for.

WHY THIS IS NOT `wget`

Three ways a download is wrong, and only the first is what people check:

  1. **Short.** A 987 MB fragment of a 5.16 GB GGUF. It exists, it is non-empty, it passes every
     `[ -s "$f" ]` test ever written, it LOADS, it serves, and it answers nonsense that gets
     recorded as model quality. This one cost real time on this project.
  2. **Complete and not a model.** An HTTP error page saves at exactly the length the server
     promised. A byte count cannot see it; the first four bytes can.
  3. **Complete, a real model, and the WRONG ONE.** A Q8_0 downloaded where a Q4_K_M was expected
     loads without complaint and silently changes every speed and memory figure it appears in.
     This is the one nothing catches by accident, and the one that matters for a comparison.

So a fetch here is not done when the bytes stop arriving. It is done when the file has been read
back and has agreed with what was asked for.

WHERE THE CREDENTIAL LIVES

A token is read from the environment, or from the Hub's own stored login, and never from an
argument. An argument is visible in `ps` to every user on the machine and lands in shell history;
the flag exists because some environments have no other route, and its help says so.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sys
from pathlib import Path

from . import gguf_io

#: Hosts where plain HTTP is acceptable. Loopback only, and this is a security model rather than a
#: convenience: the guard exists because anything on the network path can substitute weights in
#: transit, and there is no network path to loopback. Browsers treat localhost as a secure context
#: for exactly this reason. It also means a local mirror or a test server does not need a
#: certificate, which keeps the guard from being the thing people switch off.
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]", "::1")


def _is_loopback(url):
    host = url.split("://", 1)[-1].split("/", 1)[0].rsplit("@", 1)[-1]
    host = host.rsplit(":", 1)[0] if host.count(":") == 1 else host
    return host in LOOPBACK_HOSTS


#: Read in chunks this size when hashing. Large enough that a 5 GB file is not a million syscalls,
#: small enough that memory use does not scale with the model.
CHUNK = 1 << 22


class FetchError(Exception):
    """A download that cannot be trusted, phrased for a person."""


def build_parser():
    ap = argparse.ArgumentParser(
        prog="senbonzakura fetch",
        description="Download a model file and verify it is what was asked for.")
    ap.add_argument("source",
                    help="a Hub reference as repo_id:filename (e.g. "
                         "LiquidAI/LFM2.5-8B-A1B-GGUF:LFM2.5-8B-A1B-Q4_K_M.gguf), or a https URL")
    ap.add_argument("--out", default=".", help="directory to place the file in (default: here)")
    ap.add_argument("--revision", default=None,
                    help="a Hub revision. Pass a commit sha for a fetch that is reproducible: a "
                         "branch name resolves to different bytes on different days")
    ap.add_argument("--expect-size", type=int, default=None,
                    help="exact byte count. Checked before anything else looks at the contents")
    ap.add_argument("--expect-sha256", default=None,
                    help="hex digest the file must have. The only check that is proof rather than "
                         "evidence, when you have a digest from a trustworthy place")
    ap.add_argument("--expect-quant", default=None,
                    help="quantisation the GGUF must record (default: whatever its filename "
                         "claims; pass 'none' to skip)")
    ap.add_argument("--expect-arch", default=None,
                    help="architecture the GGUF must record, e.g. lfm2")
    ap.add_argument("--no-verify-gguf", action="store_true",
                    help="skip the header checks. For a file that is not a GGUF at all")
    ap.add_argument("--hf-token", default=None,
                    help="Hub token for a gated or private repo. PREFER $HF_TOKEN or a stored "
                         "`hf auth login`: an argument is visible in `ps` and in shell history")
    ap.add_argument("--force", action="store_true", help="re-download over an existing file")
    return ap


def parse_source(source):
    """`repo:file`, or a URL. Returns ("hub", repo, file) or ("url", url, filename)."""
    if source.startswith("http://") and _is_loopback(source):
        name = source.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0]
        if not name:
            raise FetchError(f"could not work out a filename from {source}")
        return "url", source, name
    if source.startswith("http://"):
        # Refused rather than upgraded. Fetching model weights over plain HTTP means anything on
        # the path can substitute them, and a substituted model is undetectable by every check in
        # this file: it would be a complete, valid GGUF of the right architecture and quant.
        # Silently rewriting the scheme would also hide that the caller asked for something unsafe.
        raise FetchError(
            f"{source} is plain HTTP. Model weights fetched over HTTP can be replaced in transit "
            f"by anything on the path, and the replacement would pass every check here. Use https.")
    if source.startswith("https://"):
        name = source.rstrip("/").rsplit("/", 1)[-1].split("?", 1)[0]
        if not name:
            raise FetchError(f"could not work out a filename from {source}")
        return "url", source, name
    # A Windows path like C:\x would split on the same colon, so require a slash before it: a Hub
    # id always has an owner, and no drive letter does.
    if ":" in source and "/" in source.split(":", 1)[0]:
        repo, _, filename = source.partition(":")
        if not filename:
            raise FetchError(
                f"{source} names a repository and no file. Give it as repo_id:filename, e.g. "
                f"LiquidAI/LFM2.5-8B-A1B-GGUF:LFM2.5-8B-A1B-Q4_K_M.gguf")
        return "hub", repo, filename
    raise FetchError(
        f"could not read {source!r} as a source. Use repo_id:filename for the Hub (the colon "
        f"separates them) or a full https URL.")


def sha256_of(path, *, chunk=CHUNK):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def resolve_token(explicit=None):
    """A token from the argument, then the environment, then whatever the Hub has stored.

    Deliberately in that order and deliberately never logged. The Hub's own store is last only
    because an explicit choice should win, not because it is worse; it is in fact the best of the
    three, since the credential never enters this process's arguments or environment.
    """
    if explicit:
        return explicit
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env
    return None          # huggingface_hub falls back to its own stored login on its own


def verify(path, *, expect_size=None, expect_sha256=None, expect_quant=None, expect_arch=None,
           check_gguf=True, log=print):
    """Every check, cheapest first, and each one says what it rules out.

    Order is not arbitrary. Size is one `stat`. The header is a few kilobytes. The digest reads the
    whole file, so it goes last: there is no sense hashing five gigabytes to then discover the
    first four bytes say HTML.
    """
    p = Path(path)
    actual = p.stat().st_size

    if expect_size is not None and actual != expect_size:
        raise FetchError(
            f"{p.name} is {actual:,} bytes and {expect_size:,} were expected"
            f"{' (short by ' + format(expect_size - actual, ',') + ')' if actual < expect_size else ''}"
            f". A short file of this kind still loads and still serves, so nothing downstream "
            f"would have told you.")
    log(f"  size: {actual:,} bytes" + ("" if expect_size is None else " (as expected)"))

    head = None
    if check_gguf:
        want = None if expect_quant in (None, "none") else expect_quant
        try:
            head = gguf_io.verify(
                p,
                expect_quant=False if expect_quant == "none" else want,
                expect_arch=expect_arch)
        except gguf_io.GGUFError as e:
            raise FetchError(f"{p.name} downloaded but is not the GGUF it should be: {e}") from e
        log(f"  header: {head['file_type']}, architecture {head['architecture']}, "
            f"{head['tensor_count']} tensors")

    if expect_sha256:
        got = sha256_of(p)
        if got != expect_sha256.lower():
            raise FetchError(
                f"{p.name} has sha256 {got}, not the expected {expect_sha256.lower()}. The bytes "
                f"are not the bytes that digest describes; do not use this file.")
        log(f"  sha256: {got} (as expected)")
    return head


def download(kind, ref, filename, out_dir, *, revision=None, token=None, log=print):
    """Place the file in `out_dir` and return its path. Resumable, and never leaves a partial."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if kind == "hub":
        # huggingface_hub owns the transfer: it resumes, it checks its own etag, and it applies a
        # stored login without the token passing through here. Reimplementing that would be worse
        # in every respect, and it arrives already as a dependency of `datasets`.
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError
        try:
            got = hf_hub_download(repo_id=ref, filename=filename, revision=revision,
                                  local_dir=str(out_dir), token=token)
        except HfHubHTTPError as e:
            raise FetchError(
                f"the Hub refused {ref}:{filename}: {e}. A gated or private repository needs a "
                f"token; set $HF_TOKEN or run `hf auth login` rather than passing one as an "
                f"argument.") from e
        except (OSError, ValueError) as e:
            raise FetchError(f"could not fetch {ref}:{filename}: {e}") from e
        return Path(got)

    import urllib.error
    import urllib.request
    dest = out_dir / filename
    tmp = dest.with_name(dest.name + ".part")
    log(f"  fetching {ref}")
    # Re-checked at the point of use, not only where it was parsed. `parse_source` is the only
    # caller today and a second one would not inherit its guarantee.
    if not (str(ref).startswith("https://")
            or (str(ref).startswith("http://") and _is_loopback(str(ref)))):
        raise FetchError(
            f"refusing to fetch {ref}: https is required, except to loopback where there is no "
            f"network path to intercept")
    try:
        # The scheme audit below is suppressed, and it is the right rule: it is satisfied twice
        # over rather than waved away. `parse_source` refuses anything but https, and the guard
        # directly above re-checks it at the point of use, so a second caller cannot inherit the
        # first one's guarantee by accident. A `file:` or custom scheme cannot reach here.
        req = urllib.request.Request(ref, headers={"User-Agent": "senbonzakura-fetch"})  # noqa: S310
        with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:  # noqa: S310
            while chunk := r.read(CHUNK):
                f.write(chunk)
    except (urllib.error.URLError, OSError) as e:
        tmp.unlink(missing_ok=True)
        # HTTPError is a file-like response object as well as an exception, so it holds a socket
        # until it is closed. The suite runs with -W error::ResourceWarning precisely to turn that
        # class of leak into a failure, and it did.
        with contextlib.suppress(AttributeError, OSError):
            e.close()
        raise FetchError(f"could not download {ref}: {e}") from e
    except BaseException:
        # An interrupted download must not leave a file the next run mistakes for a finished one.
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(dest)
    return dest


def run(argv=None, log=print):
    a = build_parser().parse_args(argv)
    try:
        kind, ref, filename = parse_source(a.source)
    except FetchError as e:
        raise SystemExit(str(e)) from e

    dest = Path(a.out) / filename
    if dest.exists() and not a.force:
        log(f"{filename} is already here; verifying it rather than downloading it again "
            f"(--force to re-fetch)")
        path = dest
    else:
        try:
            path = download(kind, ref, filename, a.out, revision=a.revision,
                            token=resolve_token(a.hf_token), log=log)
        except FetchError as e:
            raise SystemExit(str(e)) from e

    log(f"verifying {path.name}")
    try:
        verify(path, expect_size=a.expect_size, expect_sha256=a.expect_sha256,
               expect_quant=a.expect_quant, expect_arch=a.expect_arch,
               check_gguf=not a.no_verify_gguf and path.suffix == ".gguf", log=log)
    except FetchError as e:
        # The file is KEPT. What is wrong with it is the interesting part, and deleting the
        # evidence on the way out is how the same wrong download happens twice.
        raise SystemExit(f"{e}\nThe file is at {path} for inspection.") from e

    log(f"{path.name}: verified")
    return 0


def main(argv=None):
    return run(argv)


if __name__ == "__main__":
    sys.exit(main())
