# Releasing

The steps that are not obvious from the repository, in the order they have to happen.

## Before anything else: pack the evaluation track

**The wheel ships an evaluation track. The repository does not.** `.gitignore` excludes `data/`,
deliberately: a corpus committed to a public repository is in its history permanently, and no
later commit takes it back out. So the blob is built at release time from a track you hold.

```sh
python tools/pack_track.py --track ~/track-heldout
python tools/pack_track.py --manifest-only        # what the packed blob says it is
```

The packer refuses a directory that is not a track (missing partitions, or no `track.json`), and
refuses to write a blob that does not survive its own round trip.

### Check you packed the corpus you meant to

The blob is **not byte-reproducible**: a fresh salt and nonce are drawn on every pack, so two
packs of the same track differ and diffing the file against the last release proves nothing.
Compare the manifest's `sha256_of_tar` instead, which is taken over the corpus and not over the
wrapping.

```sh
python tools/pack_track.py --manifest-only | grep sha256_of_tar
```

The current track is the repaired three-way split: 259 / 4,636 / 4,982. A test asserts those
counts, so a repack of a different corpus fails the suite rather than shipping quietly under the
same name. That test exists because an export sat on HuggingFace labelled "clean" for six weeks
while 94.5% of its evaluation rows were inside its own fitting set.

### The gate that stops a release forgetting

Every test touching the packed blob **skips** when there is no blob, because a source checkout
and every ordinary CI job run without one. Skipping quietly is the hazard rather than the skip
itself: a release that forgot to pack would go out with no corpus and a green suite.

So the release build sets an environment variable that turns those skips into failures:

```sh
SENBON_REQUIRE_BUNDLED=1 python -m pytest
```

Run that **after** packing and **before** building the wheel. A skip is a statement about a
source checkout; in a release it is a defect.

## Build and check the wheel

```sh
python -m build
python -c "import zipfile,sys; print([n for n in zipfile.ZipFile(sys.argv[1]).namelist() if n.endswith('.bin')])" dist/*.whl
```

The second line is not ceremony. `package-data` in `pyproject.toml` is what puts the blob in the
wheel, and a packaging change that drops it produces a wheel that installs, imports, runs
everything except `--track default`, and fails only for users.

## The README banner resolves against `main`, not against your branch

`readme = "README.md"` in `pyproject.toml`, so the README is what PyPI renders, and PyPI cannot
resolve a relative image path. The banner is therefore an absolute
`raw.githubusercontent.com/.../main/assets/brand/...` URL, exactly as the previous hero image was.

**That means the banner shows as a broken image on any branch until `main` carries the file.**
It is not broken; it is pointing at a commit `main` does not have yet. Promote `main` BEFORE
publishing to PyPI and it renders correctly in both places. Publish to PyPI first and the
package page ships with a broken image that cannot be fixed without a new release.

## The codename

Every release tag carries one, chosen by the operator, and `pre-push` refuses an annotated `v*`
tag whose message lacks a `Codename:` line. Ask; never invent one.

It goes in three places: the annotated tag message, the GitHub Release title
(`vX.Y.Z — <Codename>`), and the CHANGELOG entry header.

## The panel

A promotion to the release branch needs a multi-persona review artefact covering the promoted
commits, under `private/reviews/YYYY-MM-DD-panel-<cluster>.md`. The push path refuses without
one.

## Order

1. Pack the track.
2. `SENBON_REQUIRE_BUNDLED=1 python -m pytest`, plus lint.
3. Build the wheel; confirm the blob is inside it.
3a. Fast-forward `main` **before** the PyPI step, or the README banner ships broken.
4. CHANGELOG entry, with the codename in the header.
5. Panel artefact covering the range.
6. Annotated tag with the `Codename:` line.
7. Push, publish, paste the CHANGELOG into the GitHub Release.
