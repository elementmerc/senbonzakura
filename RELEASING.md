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

**Two wheels are built, not one.** This is the step the runbook used to leave out, and following
it literally landed the operator at the verify step with a public GitHub Release and one wheel PyPI
will not take.

The reason is that `setup.py` refuses to let a wheel carrying `vendor/bin/` claim `py3-none-any`,
and correctly tags it `linux_x86_64` instead, which PyPI does not accept. So:

- the **universal** wheel (`py3-none-any`) goes to PyPI. It is built with the vendored binaries
  moved aside, and `--track default` still resolves from it;
- the **platform** wheel (`linux_x86_64`) is attached to the GitHub Release, where a platform tag
  is not a problem;
- the **sdist** goes to both, and carries no binaries at all (`MANIFEST.in` prunes them).

```sh
python tools/build_corpora.py          # writes src/senbonzakura/data/corpora.bin
python tools/pack_track.py             # writes src/senbonzakura/data/default-track.bin

# 1. the PLATFORM wheel, for the GitHub Release
python -m build --wheel
python tools/check_wheel.py dist/*.whl                 # tag-versus-contents only

# 2. the UNIVERSAL wheel plus the sdist, for PyPI.
#    `build/` must go too: a stale staging directory carries the binaries back in even after
#    the source tree has none, which is the trap setup.py documents.
mv src/senbonzakura/vendor/bin /tmp/senbon-vendor-bin
rm -rf build
python -m build --outdir dist-pypi
mv /tmp/senbon-vendor-bin src/senbonzakura/vendor/bin

python tools/check_wheel.py dist-pypi/*.whl --release  # the artefact strangers get
python tools/check_wheel.py dist-pypi/*.tar.gz         # nothing platform-specific in the sdist
python tools/check_cuda_channels.py     # the channels `senbonzakura setup` recommends still exist
```

`--release` is no longer the part that is easy to skip, because it is no longer skippable:
`check_wheel.py` reads the wheel's own version and turns the release checks on by itself for any
wheel naming this project at a release version. Pass the flag anyway, so the intent is on the page,
but a forgotten flag can no longer let a hollow wheel through. A flag can be forgotten; a version
cannot.

The reason this changed: the wheel published as 0.3.0 carries neither blob, and `--track default`
therefore fails for everyone who installed it. The two `.bin` files are
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

## The README banner is pinned to a commit, and must stay pinned

`readme = "README.md"` in `pyproject.toml`, so the README is what PyPI renders, and PyPI cannot
resolve a relative image path. The banner is therefore an absolute
`raw.githubusercontent.com/...` URL.

**It is pinned to a commit SHA, not to a branch, and that is not a style preference.** This
section used to say "promote `main` before the PyPI step or the banner ships broken", which
treated a permanent fault as a release-ordering problem. `origin/main` carries no `assets/`
directory at all and is hundreds of commits behind `dev`, so the branch-pinned URL was 404ing
continuously, on GitHub and on the PyPI project page, for as long as the assets existed.

A branch name in an asset URL is a promise that a branch will always carry that file. A commit
SHA names content that cannot move. `tests/test_readme_images_resolve.py` refuses a branch-pinned
URL, a SHA no commit has, and a commit whose tree lacks the file, so this cannot regress quietly.

If the artwork changes, move the pin to the commit that carries the new file and let the test
check it. Do not point it back at a branch.

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

## Publishing to PyPI

**No API token, anywhere.** `.github/workflows/publish.yml` uploads with trusted publishing: PyPI
verifies a short-lived identity GitHub mints for that one workflow in this one repository, so
there is nothing to leak from a laptop, a shell history, or a repository secret. 0.3.0 went out
from a laptop and nothing records how, by whom, or from which commit.

**The workflow does not build the wheel, and cannot.** The track is not in this repository on
purpose, so a runner holding only this repository cannot produce a release artefact. Making it
able to would mean putting the harmful prompts into GitHub, which is the decision the exclusion
exists to prevent. So the build stays local, the artefacts are attached to the GitHub Release,
and the workflow verifies and uploads them.

What it verifies before anything leaves the runner: `check_wheel --release`, `twine check`, and
that the version in the filenames is the version the tag names.

### One-time setup, on PyPI, by the operator

Add a **pending publisher** under the project's publishing settings (or the account's, for a
project that does not exist yet):

| Field | Value |
|---|---|
| PyPI project name | `senbonzakura` |
| Owner | `elementmerc` |
| Repository name | `senbonzakura` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Then, in the repository settings, create two environments, `pypi` and `testpypi`, and add
yourself as a required reviewer on `pypi`. That is the second pair of eyes on an upload that
cannot be undone: a filename on PyPI can never be reused and a version number can never be
replayed. Repeat the publisher entry on TestPyPI with environment `testpypi` to rehearse.

### Publishing

Attach the **platform** wheel (`dist/*.whl`) and the sdist to the GitHub Release, and let the
workflow upload `dist-pypi/` to PyPI. Attaching the platform wheel to the Release and uploading the
universal one is the whole point of building both; swapping them puts an artefact on PyPI that
`twine` refuses after the Release is already public and the version number is spent. Publishing the release starts the
workflow; approving the `pypi` environment lets the upload run. To rehearse first, or to re-run
after a failure, use the workflow's manual trigger with the tag and `testpypi`.

## Order

1. Pack the track.
2. `python tools/check_vendor_pins.py`; refresh any pin it reports as `DUE`.
3. `SENBON_REQUIRE_BUNDLED=1 python -m pytest`, plus lint.
4. Build BOTH wheels and the sdist, per the section above; confirm the blob is inside each
   wheel, and that `check_wheel.py` passes on the universal wheel with `--release` and on the
   sdist.
4a. Fast-forward `main` **before** the PyPI step, for the documentation links. The docs site
   deploys from `main` alone, so uploading to PyPI first publishes a project page whose links
   404 until `main` moves. The deploy workflow now fetches four of those URLs from the published
   site and fails if any is not a 200, so a wrong order is caught rather than met by a reader.
   The banner no longer depends on this step: it is pinned to a commit, per the section above.
5. CHANGELOG entry, with the codename in the header.
6. Panel artefact covering the range.
7. Annotated tag with the `Codename:` line.
8. Push, then a GitHub Release with the CHANGELOG entry as its body and the built artefacts
   attached. Publishing it runs the upload workflow; approve the `pypi` environment.

---

## Two distributions, not one

Since decision Q-29 this repository publishes **two** packages, and a release means both.

| Distribution | What it is | Dependencies |
|---|---|---|
| `senbonzakura` | the abliterator and its instruments | torch, transformers, accelerate, optuna and the rest; most of a gigabyte |
| `senbonzakura-check` | `senbonzakura check`, the result-file checker | **none at all** |

The small one is built from `checker/`, owns the `senbonzakura_check` import package, and is a
DEPENDENCY of the big one. That direction is the whole arrangement: `senbonzakura check` works
from a full install because the big distribution depends on the small one, and
`pip install senbonzakura-check` costs seconds because the small one depends on nothing.

```sh
python -m build --wheel --outdir dist-checker checker/
python -m build --sdist --outdir dist-checker checker/
```

**Three things that have to hold, and all three are gated rather than remembered.**

The two versions must agree. They are separate `_version.py` files, because reading one from the
other would be an upward import for the sake of a string, so `tests/test_second_distribution.py`
asserts the agreement instead. Bump both.

Nothing under `senbonzakura_check/` may import `senbonzakura`. An upward import works perfectly
here, where both are installed, and breaks only for the person who installed the checker alone,
which is the one person this distribution exists for and the one who never appears in our CI.

The checker's dependency list stays empty. The `checker` CI job builds the wheel, installs it
into an empty environment WITHOUT `--no-deps`, and reads back what pip resolved. Measured
2026-09-11: `['pip', 'senbonzakura-check']`, a 13 MB virtualenv.

**Order.** Publish `senbonzakura-check` FIRST. The big distribution depends on it by name, so
uploading them the other way round leaves a window in which `pip install senbonzakura` cannot
resolve, and a version number on PyPI can never be replayed.

---

## A dev-only cut, which is not a release

Sometimes the thing needed is not a release but an artefact: a wheel someone can install on a
machine this one cannot reach, so the tool gets exercised as an installed package rather than as
a checkout. Every real end-to-end finding this project has had came from that, and the CI smoke
cannot supply it, because it runs from a source tree.

**It is not a release, and the difference is the whole point.** No tag, no codename, no CHANGELOG
entry, no GitHub Release, no PyPI, no TestPyPI. Section 22's codename rule and the panel gate
both attach to a `v*` tag and to the promotion of `main`; a dev cut touches neither, which is
exactly why it must not acquire a tag out of convenience.

**It does not go on any public host, and that is a security decision rather than tidiness.** The
wheel carries the packed evaluation track and six research corpora, roughly 6,500 harmful
prompts. The README discloses that for the published release and the operator has weighed it
there. A dev artefact has had no such weighing, so it moves point to point, over ssh, to a
machine that is going to use it, and nowhere else.

### Cutting one

```sh
# 1. Give it a version nobody can confuse with another cut. BOTH files, and the suite fails if
#    they disagree.
#    PEP 440 dev releases sort below the release: 0.4.0.dev1 < 0.4.0. Bump the number every
#    time. Two different wheels sharing a version is how a tester ends up reporting a bug
#    against a build nobody can identify, and pip's cache will happily reuse the older one.
$EDITOR src/senbonzakura/_version.py
$EDITOR checker/src/senbonzakura_check/_version.py

# 2. The blobs must be present, or --track default fails for the tester and for nobody here.
python tools/pack_track.py --track <your held-out track>   # if src/senbonzakura/data/ is empty
python tools/build_corpora.py

# 3. The gate that turns "skipped for want of a blob" into a failure.
SENBON_REQUIRE_BUNDLED=1 python -m pytest
ruff check src/ tests/ tools/

# 4. The PLATFORM wheel, with the vendored binaries in it. A dev cut wants this one and not the
#    universal one: `convert` and `quantise` need llama.cpp, and PyPI's tag restriction is the
#    only reason the universal wheel exists.
rm -rf build
python -m build --wheel
python tools/check_wheel.py dist/senbonzakura-*.whl

# 5. THE CHECKER'S WHEEL, which is not optional on a dev cut. `senbonzakura` declares a
#    dependency on `senbonzakura-check`, and that name is not on PyPI, so a tester handed only
#    the big wheel gets "No matching distribution found" and cannot install anything at all.
#    Both wheels travel together until the name is published.
python -m build --wheel --outdir dist-checker checker/

# 6. Hand BOTH over, and install the small one first: pip resolves the dependency at install
#    time and will go to the index for it if it is not already present.
scp dist/senbonzakura-<version>-*.whl dist-checker/senbonzakura_check-<version>-*.whl <host>:
#    On the far machine:
#      pip install --no-deps senbonzakura_check-<version>-py3-none-any.whl
#      pip install senbonzakura-<version>-*.whl
```

`check_wheel.py` turns its release checks on by itself for a wheel naming this project at a
**release** version, and a `.devN` version is not one, so the release checks stay off. Run it
anyway: the tag-versus-contents check is what catches a wheel claiming `py3-none-any` while
carrying Linux binaries, and that is independent of whether anyone is publishing.

### What to record, and where

A dev cut leaves no tag, so the only record of what someone installed is the one you write. Put
the version, the commit SHA it was built from, the `sha256` of the wheel, and who it went to in
the session handoff and in `private/decisions.md` if anything was decided by it. A bug report
against an unidentifiable build costs more than the cut saved.

### Vendored pins on a dev cut

`check_vendor_pins.py` reporting `DUE` does not block a dev cut, and refreshing for one is a
judgement call rather than a rule. Refreshing means the tester exercises what will ship; not
refreshing means the tester exercises one variable fewer. Decide it out loud and write down which
way you went, because "the production test passed" means different things under the two.
