# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""The model card reads the artefact the run actually writes.

FOUND BY A HOSTILE OUTSIDE REVIEW, 2026-09-17, from an installed wheel with no source access.
The reviewer generated a card from a real run and got:

    ## Refusal and coherence
    - KL drift: **0.014106562361121178**
    - broken output: **0.0**

    ## Corpus
    **NOT MEASURED.** Nothing in the supplied artefacts covers this.

The run writes `baseline_refusals` and `post_bake_refusals`; the card read `baseline_refusal`
and `post_bake_refusal`. `post_bake_kl` and `post_bake_broken` happened to match, which is why
two of the four rows appeared and the mismatch looked like a measurement that had not been taken.
The corpus section failed identically, reading `corpus_sha256` against a written `track_digest`.

Two things make this worse than an ordinary typo. The card is the artefact that travels with
published weights, and the missing rows are the refusal delta, which is the whole claim. And
NOT_MEASURED is a positive assertion: the card's own prose defines it as "nothing in the
supplied artefacts covers this", so a card generated this way stated something false about an
artefact sitting right next to it.

The gate has to cover BOTH ends. A fixture alone goes stale in exactly the way being fixed here:
rename a key on the writing side and a test carrying the old names keeps passing while real
cards quietly empty out. So the names are also asserted against the writer.
"""
import json
import re
from pathlib import Path

import pytest

from senbonzakura import modelcard

SRC = Path(modelcard.__file__).parent

#: Exactly as `cli.py` writes them into `abliteration.json`.
WRITTEN = {
    "model": "Qwen/Qwen3-1.7B",
    "track": "default",
    "track_digest": "sha256:abc123",
    "dir_prompts": 256,
    "baseline_refusals": 0.094,
    "post_bake_refusals": 0.0,
    "post_bake_kl": 0.0141,
    "post_bake_broken": 0.0,
    "method": "searched",
    "matched_scoring": False,
}


def _section(card, heading):
    """The body under one `## heading`, as text."""
    body = card.split(f"## {heading}", 1)[1]
    return body.split("\n## ", 1)[0]


@pytest.fixture
def card():
    return "\n".join(modelcard.build(abl=dict(WRITTEN), licence="apache-2.0"))


class TestTheNumbersTheCardExistsToCarry:

    def test_the_refusal_before_reaches_the_card(self, card):
        assert "0.094" in card, (
            "the refusal rate before the edit is missing from the card. It is in the artefact "
            "under 'baseline_refusals'.")

    def test_the_refusal_after_reaches_the_card(self, card):
        assert "refusal before" in card and "after" in card

    def test_the_refusal_section_is_not_declared_unmeasured(self, card):
        assert modelcard.NOT_MEASURED not in _section(card, "Refusal and coherence")

    def test_the_corpus_section_is_not_declared_unmeasured(self, card):
        """The digest was written under `track_digest` and read under `corpus_sha256`."""
        assert modelcard.NOT_MEASURED not in _section(card, "Corpus")

    def test_the_corpus_digest_reaches_the_card(self, card):
        assert "sha256:abc123" in card

    def test_an_artefact_with_nothing_in_it_still_says_not_measured(self):
        """The fix must not turn an honestly empty section into a silently absent one."""
        card = "\n".join(modelcard.build(abl={"model": "m"}, licence="other"))
        assert modelcard.NOT_MEASURED in _section(card, "Refusal and coherence")

    def test_the_older_singular_spelling_is_still_read(self):
        """Cards are generated from artefacts already on disk. A rename must not blank them."""
        old = {"model": "m", "baseline_refusal": 0.5, "post_bake_refusal": 0.1}
        card = "\n".join(modelcard.build(abl=old, licence="other"))
        assert "0.5" in card and "0.1" in card


class TestTheTwoEndsOfTheContractStillAgree:
    """The half a fixture cannot check: that the writer still writes these names.

    Reading the writer's source for a literal key string is unusual in a test and is deliberate.
    The defect was a disagreement between two files, and a test that only exercises one of them
    cannot see a disagreement.
    """

    @pytest.mark.parametrize("key", ["baseline_refusals", "post_bake_refusals", "post_bake_kl",
                                     "post_bake_broken", "track_digest"])
    def test_the_run_still_writes_the_key_the_card_reads(self, key):
        written = (SRC / "cli.py").read_text(encoding="utf-8")
        assert re.search(rf'"{key}"\s*:', written), (
            f"`cli.py` no longer writes {key!r} into the abliteration artefact, and "
            f"`modelcard.py` still reads it. A card built from a real run will drop that row and "
            f"say NOT MEASURED, which is what this test exists to stop happening a second time.")

    @pytest.mark.parametrize("key", ["baseline_refusals", "post_bake_refusals", "track_digest"])
    def test_the_card_still_reads_the_key_the_run_writes(self, key):
        read = (SRC / "modelcard.py").read_text(encoding="utf-8")
        assert f'"{key}"' in read, (
            f"`cli.py` writes {key!r} and `modelcard.py` no longer looks for it.")


class TestTheOneCorpusTwoStrangersCanShare:
    """`--track default` recorded no digest, which is the wrong way round.

    FOUND BY A HOSTILE OUTSIDE REVIEW, 2026-09-17: an artefact from a bundled-track run carried
    `track_digest: null`. `--track default` is an alias, not a path, so the digest helper asked
    `Path("default").is_dir()`, got false, and recorded nothing.

    The bundled track is the ONE corpus that is byte-identical between two strangers, so its
    digest is the only one a reader can use to confirm two runs scored the same rows. The paths
    that did get a digest were private local directories nobody else can resolve.

    Guarded by availability rather than asserted flat. A source checkout carries no packed track
    until it is built, and CI skips the artefact-fetch step on some rows, so an unconditional
    assertion here is green on a developer machine and red on every platform that matters. That
    exact mistake put CI red for three days in September.
    """

    @pytest.fixture(autouse=True)
    def _needs_the_bundled_track(self):
        from senbonzakura.bundled import cache_dir
        if not cache_dir().is_dir():
            pytest.skip("no unpacked bundled track on this machine")

    def test_the_bundled_alias_produces_a_digest(self):
        from senbonzakura import cli
        digests = cli._track_digests("default")
        assert digests, "`--track default` recorded no corpus digest"
        assert "bad_eval_ds" in digests, f"no digest for the scored split: {sorted(digests)}"

    def test_a_path_that_is_not_there_still_records_nothing(self):
        """The honest answer stays the honest answer. No fabricated digest."""
        from senbonzakura import cli
        assert cli._track_digests("definitely-not-a-track") is None

    def test_the_digest_reaches_the_card(self):
        from senbonzakura import cli
        abl = dict(WRITTEN, track="default", track_digest=cli._track_digests("default"))
        card = "\n".join(modelcard.build(abl=abl, licence="apache-2.0"))
        assert modelcard.NOT_MEASURED not in _section(card, "Corpus")


def test_a_real_artefact_on_disk_renders_every_section(tmp_path):
    """The end to end shape of the reviewer's reproduction, minus the model."""
    artefact = tmp_path / "abliteration.json"
    artefact.write_text(json.dumps(WRITTEN), encoding="utf-8")
    abl = json.loads(artefact.read_text(encoding="utf-8"))
    card = "\n".join(modelcard.build(abl=abl, licence="apache-2.0"))
    for heading in ("Refusal and coherence", "Corpus"):
        assert modelcard.NOT_MEASURED not in _section(card, heading), (
            f"the {heading!r} section reports nothing measured, from an artefact that measures it")
