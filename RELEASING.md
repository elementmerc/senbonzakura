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

## Before the wheel leaves this machine

```sh
python tools/build_corpora.py          # writes src/senbonzakura/data/corpora.bin
python tools/pack_track.py             # writes src/senbonzakura/data/default-track.bin
python -m build --wheel
python tools/check_wheel.py dist/*.whl --release
```

`--release` is the part that is easy to skip and expensive to skip. The two `.bin` files are
generated artefacts kept out of git on purpose: they hold harmful prompts and the corpus is
published as a gated dataset. So a wheel built from a plain clone contains neither, installs
happily, imports happily, answers `--help` happily, and then fails `--track default` for every
person who installs it. Nothing about that wheel looks wrong from the outside.

This was live on 2026-09-08: CI builds from a clean checkout, so the wheel it checked had no
corpora, and the clean-room check that was supposed to catch it asked whether `doctor` printed
the words "corpus advbench". It prints those words on both outcomes.

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

## Refresh the vendored pins. Every release, not when someone remembers.

```sh
python tools/check_vendor_pins.py
```

The wheel ships third-party artefacts, and every one of them is pinned. Pinning is what makes a
build reproducible; it is also how a project comes to ship a year-old dependency with a year of
known bugs in it. Baseline Section 5's cooldown only says what **not** to adopt, so the operator's
rule of 2026-08-17 supplies the other half:

> Adopt nothing younger than the cooldown. Adopt everything that has aged past it, at every
> release.

The check reports one of four things per pin.

| | Meaning | Blocks a release |
|---|---|---|
| `ok` | The pin is the newest release that has cleared the cooldown | no |
| `DUE` | Something newer has aged past the cooldown. Refresh it now | no, but this is the step |
| `STOP` | The pin is over 90 days old | **yes** |
| `??` | The upstream could not be reached | no, and it is never read as `ok` |

`DUE` is advisory rather than blocking because the obligation is per release and releases are not
daily. `STOP` is the backstop for a pin nobody has looked at in a season, which is what
"we will do it next release" reliably becomes. `??` is a network problem, not a stale pin, and
failing on it would only teach people to skip the check.

To refresh: update `tag` and `published` in `src/senbonzakura/vendor/pins.json` to what the check
names, re-run the vendoring tool so the hashes come from files it actually downloaded, and re-run
this check until it reads `ok`. Never paste a hash from a web page: a hash nobody verified is
indistinguishable from a correct one right up to the moment it matters.

**Do this before the tests**, since a refreshed binary is a thing the suite should run against.

## The panel

A promotion to the release branch needs a multi-persona review artefact covering the promoted
commits, under `private/reviews/YYYY-MM-DD-panel-<cluster>.md`. The push path refuses without
one.

## Order

1. Pack the track.
2. `python tools/check_vendor_pins.py`; refresh any pin it reports as `DUE`.
3. `SENBON_REQUIRE_BUNDLED=1 python -m pytest`, plus lint.
4. Build the wheel; confirm the blob is inside it.
4a. Fast-forward `main` **before** the PyPI step, or the README banner ships broken.
5. CHANGELOG entry, with the codename in the header.
6. Panel artefact covering the range.
7. Annotated tag with the `Codename:` line.
8. Push, publish, paste the CHANGELOG into the GitHub Release.
