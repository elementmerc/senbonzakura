# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The evaluation track that ships inside the package, and the reasons it is wrapped.

WHAT THIS IS FOR, STATED WITHOUT FLATTERY

The prompts are encrypted with a key that ships beside them. Anyone who reads this file can
recover them, and that is not a defect: **this is not a secret, it is a speed bump.** It exists
so that a crawler scraping PyPI for harmful text finds a blob of noise instead of 4,895 harmful
prompts in a plaintext file, while a person who deliberately runs the tool gets the corpus with
no ceremony. Deliberate use, yes; automatic ingestion, no.

Do not describe this as protection. If anything ever depends on these prompts staying secret,
this is the wrong mechanism and there is no right one that ships in a public wheel.

WHY IT IS BUILT THIS WAY

The construction is BLAKE2b as a pseudo-random function in counter mode, with an HMAC-SHA256 tag
over the whole message. That is a sound stream cipher shape and it needs nothing outside the
standard library, which matters: adding a cryptography dependency to obtain a property we have
just said we do not have would be paying a real cost for a decorative one.

The tag is the part that earns its place. A truncated or corrupted blob would otherwise decrypt
to plausible-looking rubbish, and rubbish prompts produce a refusal direction fitted on noise and
a number that looks like a result. It fails loudly instead.

WHAT THE LICENCE REQUIRES, AND WHY YOU ARE TOLD ON FIRST USE

The bundled track is distributed under CC BY-NC 4.0: attribution to every upstream source, and
non-commercial use only. A user who installs a package and runs a default has not read a dataset
card, so shipping the rows without saying so would walk them into a licence breach they were
never shown. Hence `notice()`, printed the first time the bundled track is used in a process.
"""
import hashlib
import hmac
import io
import json
import os
import struct
import tarfile
from pathlib import Path

#: Container magic and version. A blob that does not start with this is not ours, and saying so
#: beats failing later with a confusing decode error.
MAGIC = b"SBZ1"

SALT_LEN = 16
NONCE_LEN = 16
TAG_LEN = 32
_BLOCK = 64

#: The key material that ships with the package. It is written here in the open, on purpose,
#: because pretending otherwise would be the only dishonest part of this file. See the module
#: docstring: this raises the cost of casual scraping, and nothing else.
PASSPHRASE = b"senbonzakura/evaluation-track/not-a-secret-by-design"

#: Where the packed track sits inside the installed package.
DATA_NAME = "default-track.bin"

#: The alias a user types instead of a path.
ALIAS = "default"

LICENCE = "CC BY-NC 4.0"

#: Whether the licence notice has been printed in this process. A module-level flag in a mutable
#: holder rather than a bare global, so the linter's objection to `global` does not have to be
#: silenced and tests can reset it without reaching into module internals.
_state = {"notified": False}


def data_path():
    return Path(__file__).resolve().parent / "data" / DATA_NAME


def is_available():
    return data_path().is_file()


def _derive(salt):
    return hashlib.blake2b(PASSPHRASE, salt=salt, digest_size=32).digest()


def _keystream(key, nonce, length):
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hashlib.blake2b(
            nonce + struct.pack("<Q", counter), key=key, digest_size=_BLOCK).digest()
        counter += 1
    return bytes(out[:length])


def _xor(data, stream):
    return bytes(a ^ b for a, b in zip(data, stream, strict=True))


def pack(payload, *, salt=None, nonce=None):
    """Wrap raw bytes into the container. `salt` and `nonce` are injectable for tests only."""
    salt = salt if salt is not None else os.urandom(SALT_LEN)
    nonce = nonce if nonce is not None else os.urandom(NONCE_LEN)
    if len(salt) != SALT_LEN or len(nonce) != NONCE_LEN:
        raise ValueError("salt and nonce must be exactly SALT_LEN and NONCE_LEN bytes")
    key = _derive(salt)
    body = _xor(payload, _keystream(key, nonce, len(payload)))
    head = MAGIC + salt + nonce + body
    return head + hmac.new(key, head, hashlib.sha256).digest()


def unpack(blob):
    """Recover the raw bytes, or raise ValueError naming what is wrong.

    Every failure here is checked before anything downstream sees a prompt. A blob that decrypts
    to rubbish would otherwise fit a refusal direction on noise and report a number for it.
    """
    if len(blob) < len(MAGIC) + SALT_LEN + NONCE_LEN + TAG_LEN:
        raise ValueError("the bundled track is too short to be a valid container; the package "
                         "install is incomplete or truncated.")
    if not blob.startswith(MAGIC):
        raise ValueError(f"the bundled track does not start with {MAGIC!r}, so it is not a "
                         f"senbonzakura container. Reinstall the package.")
    offset = len(MAGIC)
    salt = blob[offset:offset + SALT_LEN]
    nonce = blob[offset + SALT_LEN:offset + SALT_LEN + NONCE_LEN]
    body = blob[offset + SALT_LEN + NONCE_LEN:-TAG_LEN]
    tag = blob[-TAG_LEN:]
    key = _derive(salt)
    if not hmac.compare_digest(hmac.new(key, blob[:-TAG_LEN], hashlib.sha256).digest(), tag):
        raise ValueError("the bundled track failed its integrity check, so it has been truncated "
                         "or modified. Refusing to use it: corrupt prompts would fit a refusal "
                         "direction on noise and report a number for it. Reinstall the package.")
    return _xor(body, _keystream(key, nonce, len(body)))


def notice(log=print):
    """Say what the bundled track is and what its licence requires. Once per process."""
    if _state["notified"]:
        return
    _state["notified"] = True
    log(f"Using the bundled Senbonzakura evaluation track ({LICENCE}).")
    log("  Attribution is required, and use is NON-COMMERCIAL. The prompts come from AdvBench")
    log("  (Zou et al. 2023), Alpaca (Taori et al. 2023) and two HuggingFace datasets; the full")
    log("  chain is in docs/evaluation-track-card.md.")
    log("  It is wrapped in the package to keep it out of automated scrapes, not to keep it")
    log("  secret: the key ships beside it and this is a speed bump rather than protection.")
    log("  Pass --track to use your own corpus instead.")


def _reset_notice_for_tests():
    """Let a test see the notice again. The flag is per process, and tests share one."""
    _state["notified"] = False


def manifest():
    """What the packed track says it is, without unpacking the prompts."""
    blob = _read()
    raw = unpack(blob)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        try:
            member = tar.extractfile("manifest.json")
        except KeyError:
            member = None            # tarfile raises rather than returning None for an absent name
        if member is None:
            raise ValueError("the bundled track carries no manifest.json, so nothing records "
                             "which corpus it holds or what licence it is under.")
        return json.loads(member.read().decode("utf-8"))


def _read():
    path = data_path()
    if not path.is_file():
        raise ValueError(
            f"no bundled evaluation track is installed at {path}. A source checkout does not "
            f"carry one until `python tools/pack_track.py` has been run; a wheel should. Pass "
            f"--track with your own corpus, or build one with `senbonzakura track`.")
    return path.read_bytes()


def extract(destination, log=print):
    """Unpack the bundled track into `destination` and return the directory.

    Materialised on disk rather than held in memory because everything downstream expects a track
    DIRECTORY with a `track.json` beside the partitions, and inventing a second in-memory path
    through the whole tool to save one extraction would be a lot of new surface for no gain.
    """
    # The blob is read BEFORE the notice, not after. Printed first, the notice announced a
    # licensed corpus and its attribution terms, and the very next line said the corpus is not
    # installed. A user on a fresh clone got a paragraph about what they were using, followed by
    # being told they were not using it.
    raw = unpack(_read())
    notice(log=log)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        members = [m for m in tar.getmembers()
                   if not m.name.startswith(("/", "..")) and ".." not in Path(m.name).parts]
        if len(members) != len(tar.getmembers()):
            raise ValueError("the bundled track contains a path outside its own directory, which "
                             "a track this project packed never does. Refusing to extract it.")
        tar.extractall(destination, members=members)     # noqa: S202 - members filtered above
    return destination


def cache_dir():
    """Where an extracted copy lives, so it is unpacked once rather than per run."""
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "senbonzakura" / "bundled-track"


#: Names the extracted cache's provenance. Holds the sha256 of the packed data it came from, so
#: the cache can be keyed on CONTENT rather than on the fact that something is already there.
STAMP_NAME = ".packed-sha256"


def packed_digest():
    """The sha256 of the packed track shipped in this install."""
    h = hashlib.sha256()
    with open(data_path(), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_is_current(target):
    """Does the extracted cache come from the packed track this install actually carries?

    IT USED TO ASK ONLY WHETHER `track.json` EXISTED, which is a presence test rather than an
    identity one, and this project has now met that distinction three times. A release correcting
    the corpus would extract nothing, because a directory was already there, and every run
    afterwards would fit directions on the OLD prompts with nothing anywhere saying so. Prompts
    decide every measurement here, so that is not a stale cache; it is a silently wrong result.

    Content, not metadata. Soup keys its shard cache on basename, size and mtime, and notes that
    without it a checkpoint retrained in place would stream the wrong weights. Size and mtime can
    both be preserved by an edit in place, and a project whose tagline is receipts should hash what
    it actually read.
    """
    if not (target / "track.json").is_file():
        return False
    stamp = target / STAMP_NAME
    if not stamp.is_file():
        # Extracted by a build that predates the stamp. Its contents cannot be vouched for, so it
        # is re-extracted once and stamped, rather than trusted because it happens to be there.
        return False
    try:
        return stamp.read_text(encoding="utf-8").strip() == packed_digest()
    except OSError:
        return False


class BundledTrackError(Exception):
    """The bundled track is not in this install, said in a form callers can turn into a refusal.

    A `ValueError` used to come out of here and nothing between the loader and the command line
    caught it, so `--track default` on an install without the packed track was a traceback rather
    than the one-line refusal every other missing-input path produces. That install is a normal
    thing to have: the blob is a generated artefact kept out of git, so every source checkout has
    none until it is built.
    """


def ensure(log=print):
    """The bundled track as a directory on disk, extracting it the first time it is needed."""
    target = cache_dir()
    if not is_available() and not _cache_is_current(target):
        raise BundledTrackError(
            f"no bundled evaluation track is installed at {data_path()}. A source checkout does "
            f"not carry one until `python tools/pack_track.py` has been run; a wheel should. "
            f"Pass --track with your own corpus, or build one with `senbonzakura track`.")
    if _cache_is_current(target):
        notice(log=log)
        return target
    tmp = target.with_name(target.name + ".unpacking")
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    extract(tmp, log=log)
    inner = tmp / "track"
    source = inner if inner.is_dir() else tmp
    if target.exists():
        import shutil
        shutil.rmtree(target)
    source.rename(target)
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    # Stamped LAST, after everything else is in place, so an interrupted extraction leaves a cache
    # that fails the check and gets redone rather than one that passes it and is half a track.
    (target / STAMP_NAME).write_text(packed_digest() + "\n", encoding="utf-8")
    return target
