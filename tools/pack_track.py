#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Pack an evaluation track into the blob that ships inside the package.

The output is deliberately NOT reproducible byte for byte: a fresh salt and nonce are drawn each
time, so two packs of the same track differ. That is how the construction is supposed to work, and
it means the wheel's blob cannot be diffed against a previous release to confirm the corpus is
unchanged. The manifest inside carries a sha256 of the tar for exactly that reason, and
`--manifest-only` prints it without touching anything.

Usage:
    python tools/pack_track.py --track ~/track-heldout
    python tools/pack_track.py --track ~/track-heldout --out src/senbonzakura/data/default-track.bin
    python tools/pack_track.py --manifest-only
"""
import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from senbonzakura import bundled

#: Directories a packed track must carry. `good_eval_ds` and `good_matched_ds` are optional, so
#: they are copied when present and not demanded.
REQUIRED = ("bad_ds", "bad_eval_ds", "good_ds")
OPTIONAL = ("good_eval_ds", "good_matched_ds")


def build_tar(track, log=print):
    """A gzipped tar of the track, plus a manifest describing what went in.

    Deterministic within itself: members are sorted, and mtime/uid/gid/names are normalised, so
    the same track produces the same tar bytes and the manifest's hash means something.
    """
    track = Path(track)
    missing = [d for d in REQUIRED if not (track / d).is_dir()]
    if missing:
        raise SystemExit(
            f"pack-track: {track} is missing {', '.join(missing)}. That is not a track this tool "
            f"built; run `senbonzakura track --audit` against it first.")
    if not (track / "track.json").is_file():
        raise SystemExit(
            f"pack-track: {track} has no track.json, so nothing records where its partition "
            f"boundaries fell. Packing it would ship a corpus whose held-out slice cannot be "
            f"located. Rebuild it with `senbonzakura track`.")

    present = [d for d in REQUIRED + OPTIONAL if (track / d).is_dir()]
    log(f"pack-track: packing {', '.join(present)} from {track}")

    def norm(info):
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        return info

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        for name in [*sorted(present), "track.json"]:
            source = track / name
            for path in sorted(source.rglob("*")) if source.is_dir() else [source]:
                if path.is_dir():
                    continue
                arcname = str(Path("track") / path.relative_to(track))
                if path.name == "track.json":
                    # Rewritten rather than copied, so the build machine's paths never enter the
                    # tar at all. See `scrub_manifest`.
                    body = json.dumps(
                        scrub_manifest(json.loads(path.read_text(encoding="utf-8"))),
                        indent=2).encode("utf-8")
                    info = tarfile.TarInfo(arcname)
                    info.size = len(body)
                    tar.addfile(norm(info), io.BytesIO(body))
                    continue
                tar.add(path, arcname=arcname, filter=norm)
    raw = buf.getvalue()

    counts = _counts(track)
    manifest = {
        "schema": "senbonzakura-bundled/1",
        "licence": bundled.LICENCE,
        "partitions": present,
        "counts": counts,
        "sha256_of_tar": hashlib.sha256(raw).hexdigest(),
    }
    buf2 = io.BytesIO()
    with tarfile.open(fileobj=buf2, mode="w:gz", compresslevel=9) as tar:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as inner:
            for member in inner.getmembers():
                tar.addfile(member, inner.extractfile(member))
        blob = json.dumps(manifest, indent=2, sort_keys=True).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(blob)
        tar.addfile(norm(info), io.BytesIO(blob))
    return buf2.getvalue(), manifest


def scrub_manifest(doc):
    """The track manifest with the build machine's filesystem removed.

    `track.json` records where each corpus came from, which on the machine that built it is an
    absolute path. Packed as-is, that ships in every wheel and unpacks into every user's
    `~/.cache/senbonzakura/bundled-track/`: the 2026-09-10 panel read
    `/home/heph-agent/track2-enriched-backup/contrast/axis-labels-both.tsv` out of the published
    blob. Baseline 13 forbids private paths in shipped artefacts, and `74e571f` fixed the same
    class of defect one level up; this one survived because the blob predates it and was never
    repacked.

    The basename is kept because it is the only part that carries meaning to a reader (which
    corpus, not whose disk).
    """
    out = dict(doc)
    sources = out.get("sources")
    if isinstance(sources, dict):
        out["sources"] = {k: (os.path.basename(v) if isinstance(v, str) else v)
                          for k, v in sources.items()}
    return out


def _counts(track):
    try:
        from datasets import load_from_disk
    except ImportError:
        return {}
    out = {}
    for name in REQUIRED + OPTIONAL:
        p = track / name
        if p.is_dir():
            try:
                out[name] = len(load_from_disk(str(p)))
            except Exception:
                out[name] = None
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="pack_track.py",
        description="Pack an evaluation track into the blob that ships inside the package.")
    ap.add_argument("--track", help="the track directory to pack")
    ap.add_argument("--out", default=str(bundled.data_path()),
                    help="where the blob goes (default: inside the package)")
    ap.add_argument("--manifest-only", action="store_true",
                    help="print what the installed blob says it is, and change nothing")
    a = ap.parse_args(argv)

    if a.manifest_only:
        print(json.dumps(bundled.manifest(), indent=2, sort_keys=True))
        return 0
    if not a.track:
        ap.error("--track is required unless --manifest-only is given")

    raw, manifest = build_tar(a.track)
    blob = bundled.pack(raw)
    check = bundled.unpack(blob)
    if check != raw:
        raise SystemExit("pack-track: the blob did not survive its own round trip. Refusing to "
                         "write it.")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".partial")
    tmp.write_bytes(blob)
    os.replace(tmp, out)
    print(f"pack-track: wrote {out} ({len(blob):,} bytes)")
    print(f"pack-track: {json.dumps(manifest['counts'])}")
    print(f"pack-track: sha256 of the packed tar {manifest['sha256_of_tar'][:24]}...")
    print("pack-track: NOTE the blob is not byte-reproducible (fresh salt and nonce each pack). "
          "Compare releases with the manifest's sha256_of_tar, not with the file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
