# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Three surfaces that were accurate and still left the reader with a false belief.

All three found by hostile outside reviewers on 2026-09-17, working from an installed wheel with
no source access. None of them is a wrong number. Each is a true statement arranged so that the
obvious reading of it is wrong, which is the harder class and the one this project cares about,
because the whole product is a claim about a measurement.

  the artefact   `directions_per_layer` reads [3, 2, 1, 3, ...] beside `num_directions: 1`. The
                 list is what was EXTRACTED; the scalar is what was ABLATED. A reader quoting the
                 list concludes three directions were removed per layer when one was.

  the flag       `--kl-scale` does nothing under the default `--search pareto`, which never
                 evaluates the weighted sum. A reviewer ran the same seed at 4.0 and at -5 and
                 got an identical winner. Only the man page said "scalar"; `--help` did not, so a
                 user who set it and saw no change concluded it had no effect on the MODEL rather
                 than no effect on the SEARCH.

  the licence    the bundled corpus attribution printed once per PROCESS rather than once per
                 CORPUS, so `doctor` loaded all six and printed AdvBench's alone. XSTest's 450
                 CC-BY-4.0 rows were read with their attribution nowhere. The install page claims
                 the tool prints it "the first time it loads one"; it printed it the first time it
                 loaded any one.
"""
import types

import pytest

from senbonzakura import cli, corpora


class TestTheArtefactSaysWhichCountIsTheEdit:

    def test_the_note_distinguishes_extracted_from_ablated(self):
        """The distinction was absent while a paragraph explained a subtler indexing one."""
        source = (__import__("pathlib").Path(cli.__file__)).read_text(encoding="utf-8")
        note = source.split('"directions_index_note": (', 1)[1].split("),", 1)[0]
        assert "NOT THE" in note and "num_directions" in note, (
            "directions_index_note explains layer versus position indexing and still does not say "
            "that directions_per_layer is what was extracted rather than what was ablated")

    def test_the_note_still_explains_the_indexing(self):
        """The new sentence must not have displaced the one that was already there."""
        source = (__import__("pathlib").Path(cli.__file__)).read_text(encoding="utf-8")
        note = source.split('"directions_index_note": (', 1)[1].split("),", 1)[0]
        assert "residual-stream position" in note
        assert "axis_separations" in note


class TestADeadKnobSaysSo:

    def _args(self, **kw):
        base = dict(search="pareto", kl_scale=4.0)
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_setting_it_under_pareto_is_flagged(self):
        said = []
        cli._preflight_dead_knobs(self._args(kl_scale=9.0), log=said.append)
        assert said and "no effect" in said[0]

    def test_the_note_names_the_mode_that_would_honour_it(self):
        said = []
        cli._preflight_dead_knobs(self._args(kl_scale=9.0), log=said.append)
        assert "--search scalar" in said[0]

    def test_the_default_value_is_not_nagged_about(self):
        """Almost nobody sets this. Warning on a value the user never typed is noise."""
        said = []
        cli._preflight_dead_knobs(self._args(), log=said.append)
        assert not said

    def test_scalar_search_is_not_flagged(self):
        said = []
        cli._preflight_dead_knobs(self._args(search="scalar", kl_scale=9.0), log=said.append)
        assert not said

    def test_the_note_is_actually_WIRED_IN_ahead_of_the_model(self, tmp_path, monkeypatch, capsys):
        """Every other test here calls the function directly, so all of them stay green if the
        call is deleted from `run_parsed` and it never runs again. Mutation testing caught
        exactly that, on this check and on the recovery pre-flight before it. A check nothing
        calls is a check that does not exist."""
        from senbonzakura import lengthsweep

        class _Reached(Exception):
            pass

        def _boom(*_a, **_k):
            raise _Reached

        monkeypatch.setattr(cli, "Abliterator", _boom)
        monkeypatch.setattr(cli, "_preflight_device", lambda _a, log=None: "cpu")
        args = types.SimpleNamespace(
            track="default", good_ds=None, hedge_ds="", clean_ds="", harmless_matched="",
            gen_tokens=lengthsweep.DEFAULT_BUDGET, short_budget_ok=False, text_column=None,
            hf_token=None, load_in_4bit=False, model=None, out=str(tmp_path), resume=False,
            study_db=None, no_persist_study=False, search="pareto", bake_config=None,
            kl_scale=9.0)
        with pytest.raises((_Reached, SystemExit)):
            cli.run_parsed(args, None, [])
        assert "--kl-scale" in capsys.readouterr().out, (
            "the dead-knob note never reached the user, so the call site is missing")

    def test_the_help_says_where_it_applies(self):
        """The man page carried the word "scalar" and `--help` did not, which is backwards:
        `--help` is what people read."""
        from senbonzakura import parser
        text = parser.build_parser().format_help()
        assert "SCALAR objective" in text
        # The option BODY, not the usage line, which also names the flag and says nothing.
        body = text.split("  --kl-scale KL_SCALE", 1)[1][:1200]
        assert "pareto" in body, "the help never mentions the mode under which the flag is inert"


class TestEveryCorpusCarriesItsOwnAttribution:
    """A licence obligation, not a courtesy: these terms require the notice to travel."""

    @pytest.fixture(autouse=True)
    def _fresh(self):
        corpora._reset_notice_for_tests()
        yield
        corpora._reset_notice_for_tests()

    def test_a_second_corpus_is_not_silenced_by_the_first(self):
        keys = list(corpora.CORPORA)[:3]
        said = []
        for key in keys:
            corpora.notice(key, log=said.append)
        headers = [s for s in said if s.startswith("Using the bundled corpus")]
        assert len(headers) == len(keys), (
            f"{len(keys)} corpora were loaded and {len(headers)} attributions printed. The others "
            f"were used with their attribution nowhere in what the user has.")

    def test_the_same_corpus_twice_still_prints_once(self):
        """Once per corpus, not once per read. A search loads a corpus many times."""
        key = next(iter(corpora.CORPORA))
        said = []
        corpora.notice(key, log=said.append)
        corpora.notice(key, log=said.append)
        assert len([s for s in said if s.startswith("Using the bundled corpus")]) == 1

    def test_every_bundled_corpus_can_state_its_own_terms(self):
        said = []
        for key in corpora.CORPORA:
            corpora.notice(key, log=said.append)
        headers = [s for s in said if s.startswith("Using the bundled corpus")]
        assert len(headers) == len(corpora.CORPORA)
        for header in headers:
            assert "(" in header and ")" in header, f"no licence named in {header!r}"
