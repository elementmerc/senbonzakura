# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""What a mistyped argument costs, on the paths where the cost is a download or an hour.

FOUND BY ADVERSARIAL USER TESTING, 2026-09-17, from an installed wheel with no source access.
Four flags whose entire job is to be strict did not check their own arguments, and two of them
only noticed after the expensive part was already spent:

  fetch --expect-sha256 zzzznothex   transferred the whole file, then reported a mismatch
  fetch --expect-size -1             transferred the whole file, then reported a mismatch
  quantise --threads -5              quantised, verified, and reported DONE
  quantise --threads 99999           a one-second job became an open-ended hang

Every one of them is decidable from the command line before anything is fetched or started. The
shape is the same as the numeric bounds on the abliterate path (`_preflight_numbers`), found the
same way a day earlier: an argument nothing validates is an argument that fails as late as
possible, in a message about the wrong thing.

The separate `fetch` case here is a correctness bug rather than an ordering one: bare Hub repo
ids were rejected outright.
"""
import pytest

from senbonzakura import fetch, quantise
from senbonzakura.fetch import FetchError


class TestTheCanonicalHubIdsAreHubIds:
    """`gpt2` is a repo id. So are `bert-base-uncased` and `distilgpt2`.

    The parser required an owner, on the reasoning that a Windows drive letter would otherwise
    split on the same colon. A drive letter is one character and no repo id is, so length is the
    discriminator and the owner never needed to be mandatory. Rejecting the most recognisable
    ids on the Hub, and telling the user their correct syntax was wrong, is the expensive half.
    """

    @pytest.mark.parametrize("repo", ["gpt2", "bert-base-uncased", "distilgpt2", "t5-small"])
    def test_a_bare_repo_id_is_read_as_a_hub_reference(self, repo):
        assert fetch.parse_source(f"{repo}:config.json") == ("hub", repo, "config.json")

    def test_an_owner_prefixed_id_still_works(self):
        assert fetch.parse_source("openai-community/gpt2:config.json") == (
            "hub", "openai-community/gpt2", "config.json")

    @pytest.mark.parametrize("path", [r"C:\models\x.gguf", "C:/models/x.gguf"])
    def test_a_windows_drive_letter_is_still_not_a_repo(self, path):
        """The reason the rule existed. It has to keep holding."""
        with pytest.raises(FetchError):
            fetch.parse_source(path)

    def test_a_third_path_segment_is_not_a_repo_id(self):
        with pytest.raises(FetchError):
            fetch.parse_source("a/b/c:f")

    def test_the_refusal_says_the_owner_is_optional(self):
        """The old message sent people to add an owner they did not need."""
        with pytest.raises(FetchError) as e:
            fetch.parse_source("nocolon")
        assert "optional" in str(e.value)


class TestAnExpectationNothingCouldSatisfy:
    """Checked before the transfer, because the file these flags are used on is a model."""

    @pytest.mark.parametrize("digest", ["zzzznothex", "", "abc", "0" * 63, "0" * 65, "g" * 64])
    def test_a_string_that_is_not_a_sha256_is_refused(self, digest):
        with pytest.raises(SystemExit) as e:
            fetch.run(["gpt2:config.json", "--expect-sha256", digest])
        assert "--expect-sha256" in str(e.value)
        assert "64 hex" in str(e.value)

    def test_a_real_looking_digest_is_allowed_through(self):
        """The gate must not start refusing the arguments the flag exists for.

        Any digest that is 64 hex characters is well formed; whether it MATCHES is a question
        about bytes and stays where it was, after the download.
        """
        fetch._preflight_expectations(_Args(expect_sha256="A" * 64, expect_size=None))
        fetch._preflight_expectations(_Args(expect_sha256="0123456789abcdef" * 4, expect_size=None))

    def test_a_negative_byte_count_is_refused(self):
        with pytest.raises(SystemExit) as e:
            fetch.run(["gpt2:config.json", "--expect-size", "-1"])
        assert "negative" in str(e.value)

    def test_zero_bytes_is_allowed_because_a_zero_byte_file_exists(self):
        """Nothing here decides that an empty file is a mistake; the size check does that."""
        fetch._preflight_expectations(_Args(expect_sha256=None, expect_size=0))

    def test_the_refusal_says_it_cost_nothing(self):
        with pytest.raises(SystemExit) as e:
            fetch.run(["gpt2:config.json", "--expect-size", "-5"])
        assert "before the download" in str(e.value)

    def test_it_refuses_before_the_source_is_even_parsed(self):
        """A bad digest and a bad source together: the digest wins, because it is free.

        If the order flipped, a user fixing what they were told about would get a second
        refusal about something the tool already knew.
        """
        with pytest.raises(SystemExit) as e:
            fetch.run(["this is not a source", "--expect-sha256", "nope"])
        assert "--expect-sha256" in str(e.value)


class _Args:
    """A stand-in for parsed argv, so a pre-flight can be called without a parser."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class TestAWorkerCountThatIsNotAWorkerCount:
    """`llama-quantize` does not decline a thread count it cannot use.

    Measured 2026-09-17 on a 270 MB model: the default finished in 1s; `--threads 99999` was
    still on tensor 84 of 272 after 200s, in ONE thread, with 1.69 GB resident. It serialises
    and allocates. So the ceiling has to be ours, and a user watching a command that took a
    second yesterday has no way to tell a hang from slow progress.
    """

    def test_a_negative_count_is_refused(self, tmp_path):
        src = tmp_path / "m.gguf"
        src.write_bytes(b"GGUF" + b"\0" * 100)
        with pytest.raises(SystemExit) as e:
            quantise.run([str(src), "--threads", "-5"])
        assert "--threads" in str(e.value)
        assert "negative" in str(e.value)

    def test_an_impossible_count_is_refused(self, tmp_path):
        src = tmp_path / "m.gguf"
        src.write_bytes(b"GGUF" + b"\0" * 100)
        with pytest.raises(SystemExit) as e:
            quantise.run([str(src), "--threads", "99999"])
        assert "ceiling" in str(e.value)

    def test_the_refusal_explains_why_it_reads_as_a_hang(self, tmp_path):
        src = tmp_path / "m.gguf"
        src.write_bytes(b"GGUF" + b"\0" * 100)
        with pytest.raises(SystemExit) as e:
            quantise.run([str(src), "--threads", "99999"])
        assert "serialises" in str(e.value)

    def test_zero_is_still_the_documented_way_to_let_it_choose(self):
        quantise._preflight_arguments(_Args(threads=0, imatrix=None, tensor_type=[]), log=lambda *_: None)

    def test_a_count_above_the_core_count_warns_rather_than_refuses(self, monkeypatch):
        """Oversubscription is a real choice on a shared box. It is a warning, not a refusal."""
        monkeypatch.setattr(quantise.os, "cpu_count", lambda: 4)
        said = []
        quantise._preflight_arguments(_Args(threads=64, imatrix=None, tensor_type=[]), log=said.append)
        assert any("4 cores" in s for s in said)


class TestTheTwoFlagsThatExistToSaveTimeAndSpentItInstead:
    """`--bake-config` and `--resume` both did 80+ seconds of work before checking a path.

    FOUND BY ADVERSARIAL USER TESTING, 2026-09-17. `--bake-config` at a path that does not exist
    loaded the model, extracted refusal directions and began caching the KL reference before it
    looked. Its own help says it "recovers a crashed save in minutes instead of re-searching", so
    a typo in it cost exactly what it exists to avoid. `--resume` against a directory with no
    study said so correctly at 146 seconds, when the database path was derivable from `--out` at
    second zero, and the person who resumes an eight hour run has walked away by then.
    """

    def _args(self, tmp_path, **kw):
        import types
        base = dict(bake_config=None, resume=False, study_db=None, no_persist_study=False,
                    track="default", out=str(tmp_path), search="pareto")
        base.update(kw)
        return types.SimpleNamespace(**base)

    def test_a_bake_config_that_is_not_there_is_refused(self, tmp_path):
        from senbonzakura import cli
        with pytest.raises(SystemExit) as e:
            cli._preflight_recovery(self._args(tmp_path, bake_config=str(tmp_path / "nope.json")))
        assert "nothing to bake" in str(e.value)

    def test_the_refusal_says_where_the_file_comes_from(self, tmp_path):
        """A user who mistyped it needs to know what they were pointing at."""
        from senbonzakura import cli
        with pytest.raises(SystemExit) as e:
            cli._preflight_recovery(self._args(tmp_path, bake_config=str(tmp_path / "nope.json")))
        assert "best-config.json" in str(e.value)
        assert "costs nothing" in str(e.value)

    def test_a_bake_config_that_is_not_json_is_refused(self, tmp_path):
        from senbonzakura import cli
        bad = tmp_path / "bad.json"
        bad.write_text("i am not json", encoding="utf-8")
        with pytest.raises(SystemExit) as e:
            cli._preflight_recovery(self._args(tmp_path, bake_config=str(bad)))
        assert "not readable as JSON" in str(e.value)

    def test_a_real_config_passes(self, tmp_path):
        from senbonzakura import cli
        good = tmp_path / "best-config.json"
        good.write_text('{"per_layer": {}}', encoding="utf-8")
        cli._preflight_recovery(self._args(tmp_path, bake_config=str(good)))

    def test_resume_with_no_study_says_so_before_the_model(self, tmp_path):
        """A note, not a refusal. A fresh search is a legitimate thing to want."""
        from senbonzakura import cli
        said = []
        cli._preflight_recovery(self._args(tmp_path, resume=True), log=said.append)
        assert said, "--resume against an empty directory said nothing at the boundary"
        assert "FRESH" in said[0]

    def test_resume_and_no_persist_contradict_each_other(self, tmp_path):
        from senbonzakura import cli
        with pytest.raises(SystemExit) as e:
            cli._preflight_recovery(self._args(tmp_path, resume=True, no_persist_study=True))
        assert "nothing to resume from" in str(e.value)

    def test_an_existing_study_is_resumed_quietly(self, tmp_path):
        """The note must not fire on the case the flag is FOR."""
        import contextlib
        import sqlite3

        from senbonzakura import cli
        db = tmp_path / "senbon-study.db"
        # `with sqlite3.connect(...)` commits and does NOT close. Left as it was, the connection
        # survived to be finalised during a later test's gc and reported there as an unraisable
        # exception, which is a flake wearing another test's name.
        with contextlib.closing(sqlite3.connect(db)) as conn:
            conn.execute("CREATE TABLE studies (study_name TEXT)")
            conn.execute("INSERT INTO studies VALUES ('senbon-pareto')")
            conn.commit()
        said = []
        cli._preflight_recovery(self._args(tmp_path, resume=True, study_db=str(db)), log=said.append)
        assert not said, f"resuming a real study printed a fresh-search note: {said}"

    def test_the_check_is_actually_wired_in_ahead_of_the_model(self, tmp_path, monkeypatch):
        """The half that unit tests cannot see, and the half that was the whole defect.

        Every other test in this class calls `_preflight_recovery` directly, so all of them stay
        green if somebody deletes the call from `run_parsed` and the function never runs again.
        Mutation testing caught exactly that: removing the call site left this class passing.

        This asserts the ORDER the same way `test_dataset_preflight.py` does, by making the model
        constructor explode. If the pre-flight moves back below it, this test sees the explosion
        instead of the refusal and fails, which is the only thing that distinguishes a check in
        the right place from a check in the wrong place.
        """
        import types

        from senbonzakura import cli, lengthsweep

        def _never(*_a, **_k):
            raise AssertionError("the model was constructed before --bake-config was checked")

        monkeypatch.setattr(cli, "Abliterator", _never)
        monkeypatch.setattr(cli, "_preflight_device", lambda _a, log=None: "cpu")
        args = types.SimpleNamespace(
            track="default", good_ds=None, hedge_ds="", clean_ds="", harmless_matched="",
            gen_tokens=lengthsweep.DEFAULT_BUDGET, short_budget_ok=False, text_column=None,
            hf_token=None, load_in_4bit=False, model=None, out=str(tmp_path), resume=False,
            study_db=None, no_persist_study=False, search="pareto",
            bake_config=str(tmp_path / "definitely-not-here.json"))
        with pytest.raises(SystemExit, match="nothing to bake"):
            cli.run_parsed(args, None, [])

    def test_an_unreadable_database_does_not_refuse_the_run(self, tmp_path):
        """The lookup decides a SENTENCE, never what runs, so it must never block a working run."""
        from senbonzakura import cli
        db = tmp_path / "senbon-study.db"
        db.write_bytes(b"not a database at all")
        said = []
        cli._preflight_recovery(self._args(tmp_path, resume=True, study_db=str(db)), log=said.append)
        assert not said


class TestTheWarningTheSummaryUsedToSwallow:
    """llama-quantize says how many tensors the recipe could not be applied to, once, and exits 0.

    FOUND BY ADVERSARIAL USER TESTING, 2026-09-17. The binary printed

        llama_model_quantize_impl: WARNING: 180 of 272 tensor(s) required fallback quantization

    and four lines later this tool's own summary printed `verified: Q4_K_M` and stopped. Two
    thirds of the tensors were not at the requested precision and the line written to be read
    said the opposite. The warning was never lost, only unreachable: the subprocess wrote
    straight to the terminal, so nothing in this process ever saw it.

    Run against a real subprocess rather than a fake one. The thing under test IS the plumbing,
    and a mock of a pipe proves only that the code calls what it was written to call.
    """

    def _echo(self, *lines):
        import sys as _sys
        body = ";".join(f"print({line!r})" for line in lines)
        return [_sys.executable, "-c", body]

    def test_the_fallback_count_is_picked_out_of_the_stream(self):
        proc, fallback = quantise._run_quantiser(self._echo(
            "[  1/272] blk.0.attn_k.weight - converting to q4_K",
            "llama_model_quantize_impl: WARNING: 180 of 272 tensor(s) required fallback quantization",
            "llama_quantize: total time = 564.38 ms"))
        assert proc.returncode == 0
        assert fallback == (180, 272)

    def test_a_clean_run_reports_nothing_rather_than_zero(self, capsys):
        """None and 0 are different claims. One is "no tensor fell back", the other is "we did
        not see". A reader comparing two files has to be able to tell them apart.
        """
        proc, fallback = quantise._run_quantiser(self._echo("all good", "done"))
        assert proc.returncode == 0
        assert fallback is None

    def test_the_binarys_output_still_reaches_the_terminal(self, capsys):
        """A large quantisation is long and its per-tensor progress is the only sign of life it
        gives. Capturing it to read one line must not swallow the rest.
        """
        quantise._run_quantiser(self._echo("[  1/272] first", "[272/272] last"))
        out = capsys.readouterr().out
        assert "[  1/272] first" in out and "[272/272] last" in out

    def test_a_non_zero_exit_is_still_a_non_zero_exit(self):
        import sys as _sys
        proc, _ = quantise._run_quantiser([_sys.executable, "-c", "raise SystemExit(3)"])
        assert proc.returncode == 3


class TestTheRefusalArrivesBeforeTheAnnouncement:
    """A tool that prints what it is about to do and then refuses has told the user twice.

    Both of these were validated only after the header line, the binary lookup and the
    quantiser identity read. They are command-line facts and belong ahead of all three.
    """

    def test_a_missing_imatrix_is_refused_before_anything_is_printed(self, tmp_path, capsys):
        src = tmp_path / "m.gguf"
        src.write_bytes(b"GGUF" + b"\0" * 100)
        with pytest.raises(SystemExit):
            quantise.run([str(src), "--imatrix", str(tmp_path / "nope.imatrix")])
        assert "quantise" not in capsys.readouterr().out

    def test_a_malformed_tensor_type_is_refused_before_anything_is_printed(self, tmp_path, capsys):
        src = tmp_path / "m.gguf"
        src.write_bytes(b"GGUF" + b"\0" * 100)
        with pytest.raises(SystemExit):
            quantise.run([str(src), "--tensor-type", "no-equals-sign"])
        assert "quantise" not in capsys.readouterr().out
