# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""The refusal map's defaults: the shared sweep, the automatic spill, the pinned date, the fragile flag.

Each default has to give the numbers the old way gave, so the comparisons here are of whole artefacts,
byte for byte, with only the fields that record HOW the run went left out.
"""
import datetime as dt
import json
import re

import pytest
import torch
from test_attribution_hf import FAST, STUB_IDENTITY, StubModel, StubTok, _parts

from senbonzakura import argresolve, chattemplate, firsttoken
from senbonzakura import attribution as A
from senbonzakura import attribution_followups as F
from senbonzakura import attribution_hf as H
from senbonzakura import attribution_run as R

PASSED = {"passed": True, "cases": {}}
TINY_CACHE_GIB = 25000 / 2 ** 30            # about one batch of the stand-in model
BASE = ["m", "--track", "t", "--out", "o"]


def _parse(*extra):
    return R.build_parser().parse_args([*BASE, *extra])


def _stub_run(monkeypatch, out, **flags):
    monkeypatch.setattr(H, "load", lambda *_a, **_k: (StubModel(), StubTok(), dict(STUB_IDENTITY)))
    monkeypatch.setattr(H, "build_partitions", lambda *_a, **_k: (_parts(confirm=True), {"measure": {"n": 40}}))
    monkeypatch.setattr(A, "selftest", lambda **_k: PASSED)
    args = R.build_parser().parse_args(["stub", "--track", "t", "--out", str(out), "--n-measure", "40", "--batch", "8"])
    for key, value in flags.items():
        setattr(args, key, value)
    return args


def _artefact(path):
    """The artefact with the fields that only say how the run went (clock, settings) taken out."""
    doc = json.loads((path / "attribution.json").read_text())
    doc.pop("created")
    doc.pop("run_settings")
    return json.dumps(doc, sort_keys=True)


def _settings(path):
    return json.loads((path / "attribution.json").read_text())["run_settings"]


# ------------------------------------------------------------------------------------- the flags

def test_the_defaults_are_the_shared_sweep_an_automatic_spill_and_no_pinned_date():
    args = _parse()
    assert args.reference_sweep is False and args.fast_sweep is False
    assert args.prefix_cache_spill is None and args.no_prefix_cache_spill is False
    assert args.template_date is None
    assert R.spill_mode(args) == R.SPILL_AUTO


def test_the_old_fast_sweep_flag_is_still_accepted_and_the_reference_flag_is_new():
    assert _parse("--fast-sweep").fast_sweep is True
    assert _parse("--reference-sweep").reference_sweep is True


def test_the_two_sweep_flags_contradict_each_other_and_say_so(capsys):
    with pytest.raises(SystemExit):
        _parse("--fast-sweep", "--reference-sweep")
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize(("argv", "mode"), [
    ([], R.SPILL_AUTO), (["--prefix-cache-spill"], R.SPILL_ON), (["--prefix-cache-spill", "3"], R.SPILL_ON),
    (["--prefix-cache-spill", "0"], R.SPILL_OFF), (["--no-prefix-cache-spill"], R.SPILL_OFF)])
def test_the_spill_flags_pick_auto_on_or_off(argv, mode):
    assert R.spill_mode(_parse(*argv)) == mode


def test_forcing_the_spill_on_and_off_together_is_refused(capsys):
    with pytest.raises(SystemExit):
        _parse("--prefix-cache-spill", "--no-prefix-cache-spill")
    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["yesterday", "2026-1-5", "22/10/2026", "2026-13-01", "2026-02-30", "20261022", ""])
def test_a_date_that_is_not_yyyy_mm_dd_is_refused_where_it_is_read(bad, capsys):
    with pytest.raises(SystemExit):
        _parse("--template-date", bad)
    assert "--template-date" in capsys.readouterr().err


def test_a_good_date_is_kept_as_written():
    assert _parse("--template-date", "2026-10-22").template_date == "2026-10-22"
    assert argresolve.calendar_date("--template-date")("2024-02-29") == "2024-02-29"


def test_the_help_names_every_new_flag(capsys):
    with pytest.raises(SystemExit):
        R.build_parser().parse_args(["--help"])
    text = capsys.readouterr().out
    for flag in ("--reference-sweep", "--no-prefix-cache-spill", "--template-date"):
        assert flag in text


# --------------------------------------------------------------- the default sweep, bit for bit

@pytest.mark.parametrize("gib", [2.0, TINY_CACHE_GIB, 1e-6, 0.0], ids=["fits", "overflows", "holds_none", "off"])
def test_the_default_sweep_writes_the_reference_sweeps_artefact_whether_the_cache_fits_or_overflows(
        monkeypatch, tmp_path, gib):
    monkeypatch.setattr(R, "_free_host_bytes", lambda: 64 * 2 ** 30)
    default, reference = tmp_path / "default", tmp_path / "reference"
    R.run(_stub_run(monkeypatch, default, prefix_cache_gib=gib), log=lambda *_: None)
    R.run(_stub_run(monkeypatch, reference, prefix_cache_gib=gib, reference_sweep=True), log=lambda *_: None)
    assert _artefact(default) == _artefact(reference)
    assert _settings(default)["sweep"] == "fast" and _settings(reference)["sweep"] == "reference"


def test_the_artefact_says_which_sweep_ran_and_whether_the_chain_check_held(monkeypatch, tmp_path):
    lines = []
    R.run(_stub_run(monkeypatch, tmp_path), log=lines.append)
    settings = _settings(tmp_path)
    assert settings["sweep"] == "fast"
    assert set(settings["arms_computed_in_this_run"]) == {"disjoint", "purity"}
    checks = {name: arm["chain_check"] for name, arm in settings["arms_computed_in_this_run"].items()}
    # The first arm checks the chain; the second reads the chains the first built, so it has nothing to check.
    assert checks == {"disjoint": "passed", "purity": "not run"}
    assert any(line.startswith("sweep: fast (the default)") for line in lines)


def test_the_old_flag_changes_nothing(monkeypatch, tmp_path):
    plain, old = tmp_path / "plain", tmp_path / "old"
    R.run(_stub_run(monkeypatch, plain), log=lambda *_: None)
    R.run(_stub_run(monkeypatch, old, fast_sweep=True), log=lambda *_: None)
    assert _artefact(plain) == _artefact(old) and _settings(plain)["sweep"] == "fast"


def test_the_reference_sweep_says_so_in_its_log_and_shares_no_replies(monkeypatch, tmp_path):
    lines = []
    R.run(_stub_run(monkeypatch, tmp_path, reference_sweep=True), log=lines.append)
    assert any("--reference-sweep" in line for line in lines if line.startswith("sweep:"))
    arms = _settings(tmp_path)["arms_computed_in_this_run"]
    assert all(arm["replies_reused"] == 0 and arm["chains_computed"] == 0 for arm in arms.values())


# --------------------------------------------------------------------- the automatic spill

def _share(lines):
    return int(re.search(r"with (\d+)% of the batches", next(x for x in lines if x.startswith("projection:"))).group(1))


def test_the_spill_turns_itself_on_when_the_projection_says_the_cache_overflows(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "_free_host_bytes", lambda: 64 * 2 ** 30)
    lines = []
    R.run(_stub_run(monkeypatch, tmp_path, prefix_cache_gib=TINY_CACHE_GIB), log=lines.append)
    record = _settings(tmp_path)["prefix_cache"]
    assert record["spill_mode"] == "auto" and record["spill"] is True and record["spill_gib_allowed"] > 0
    assert record["projected_share_of_batches_in_main_cache"] < 1.0
    assert any(line.startswith("prefix cache spill: on automatically") for line in lines)
    served = [x for x in lines if "served from the main cache" in x]
    assert any(int(re.search(r"spill tier (\d+)", x).group(1)) > 0 for x in served), served


def test_the_spill_stays_off_when_the_projection_says_every_batch_fits(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "_free_host_bytes", lambda: 64 * 2 ** 30)
    lines = []
    R.run(_stub_run(monkeypatch, tmp_path), log=lines.append)
    record = _settings(tmp_path)["prefix_cache"]
    assert record["spill_mode"] == "auto" and record["spill"] is False and record["spill_gib_allowed"] == 0
    assert record["projected_share_of_batches_in_main_cache"] == 1.0
    assert any(line.startswith("prefix cache spill: off, because the projection says every batch fits")
               for line in lines)
    assert not any("served from the main cache" in line for line in lines)


@pytest.mark.parametrize("gib", [2.0, TINY_CACHE_GIB], ids=["fits", "overflows"])
def test_automatic_spill_gives_the_numbers_of_no_spill(monkeypatch, tmp_path, gib):
    monkeypatch.setattr(R, "_free_host_bytes", lambda: 64 * 2 ** 30)
    auto, off = tmp_path / "auto", tmp_path / "off"
    R.run(_stub_run(monkeypatch, auto, prefix_cache_gib=gib), log=lambda *_: None)
    R.run(_stub_run(monkeypatch, off, prefix_cache_gib=gib, no_prefix_cache_spill=True), log=lambda *_: None)
    assert _artefact(auto) == _artefact(off)
    assert _settings(off)["prefix_cache"]["spill_mode"] == "off" and _settings(off)["prefix_cache"]["spill"] is False


def test_forcing_the_spill_off_recomputes_what_the_cache_cannot_hold(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "_free_host_bytes", lambda: 64 * 2 ** 30)
    lines = []
    R.run(_stub_run(monkeypatch, tmp_path, prefix_cache_gib=TINY_CACHE_GIB, no_prefix_cache_spill=True),
          log=lines.append)
    assert "prefix cache spill: off, as asked" in lines
    assert _settings(tmp_path)["prefix_cache"]["spill"] is False


def test_forcing_the_spill_on_spills_even_when_everything_fits(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "_free_host_bytes", lambda: 64 * 2 ** 30)
    record = None
    R.run(_stub_run(monkeypatch, tmp_path, prefix_cache_spill=1.0), log=lambda *_: None)
    record = _settings(tmp_path)["prefix_cache"]
    assert record["spill_mode"] == "on" and record["spill"] is True


def test_the_automatic_spill_keeps_the_safety_of_the_flag_half_the_free_memory_and_no_guessing(monkeypatch):
    args = _parse()
    lines = []
    monkeypatch.setattr(R, "_free_host_bytes", lambda: 20 * 2 ** 30)
    assert R._spill_budget(args, 2 * 2 ** 30, lines.append, projected_fit=0.68) == 8 * 2 ** 30
    monkeypatch.setattr(R, "_free_host_bytes", lambda: 4 * 2 ** 30)
    assert R._spill_budget(args, 2 * 2 ** 30, lines.append, projected_fit=0.68) == 0
    assert "no room" in lines[-1]
    monkeypatch.setattr(R, "_free_host_bytes", lambda: None)
    assert R._spill_budget(args, 2 * 2 ** 30, lines.append, projected_fit=0.68) == 0
    assert "could not be measured" in lines[-1]
    assert R._spill_budget(args, 0, lines.append, projected_fit=0.5) == 0
    assert "main cache is off" in lines[-1]


def test_a_projection_of_exactly_every_batch_does_not_spill(monkeypatch):
    monkeypatch.setattr(R, "_free_host_bytes", lambda: 64 * 2 ** 30)
    assert R._spill_budget(_parse(), 2 * 2 ** 30, lambda *_: None, projected_fit=1.0) == 0
    assert R._spill_budget(_parse(), 2 * 2 ** 30, lambda *_: None, projected_fit=0.999) > 0
    assert R._spill_budget(_parse(), 2 * 2 ** 30, lambda *_: None, projected_fit=None) == 0


# ------------------------------------------------------------------------- the pinned date

DATED = ("Today: {{ strftime_now('%d %b %Y') }}|{% for m in messages %}{{ m['content'] }}{% endfor %}"
         "{% if add_generation_prompt %}>{% endif %}")
UNDATED = "{% for m in messages %}{{ m['content'] }}{% endfor %}{% if add_generation_prompt %}>{% endif %}"


class FakeClock(dt.datetime):
    """Stands in for the `datetime` that transformers' template code reads: whatever day `today` holds."""

    today = dt.datetime(2026, 10, 22, 23, 59, 0)  # noqa: DTZ001 - a naive local clock, like the one a template reads

    @classmethod
    def now(cls, tz=None):
        return cls.today if tz is None else cls.today.replace(tzinfo=tz)

    @classmethod
    def past_midnight(cls):
        cls.today = cls.today + dt.timedelta(days=1)


@pytest.fixture
def clock(monkeypatch):
    from transformers.utils import chat_template_utils
    FakeClock.today = dt.datetime(2026, 10, 22, 23, 59, 0)  # noqa: DTZ001 - a naive local clock, like the one a template reads
    monkeypatch.setattr(chat_template_utils, "datetime", FakeClock)
    return FakeClock


def _real_tokenizer(template):
    import transformers
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    chars = [chr(c) for c in range(32, 127)]
    vocab = {"[PAD]": 0, "[UNK]": 1, **{c: i + 2 for i, c in enumerate(chars)}}
    core = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))  # noqa: S106
    core.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
    return transformers.PreTrainedTokenizerFast(tokenizer_object=core, pad_token="[PAD]", unk_token="[UNK]",  # noqa: S106
                                                padding_side="left", chat_template=template)


def _real_model(seed=0):
    import transformers
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=3, num_attention_heads=2,
                                   num_key_value_heads=2, vocab_size=97, pad_token_id=0)
    return transformers.LlamaForCausalLM(cfg).eval()


class Spy:
    """A tokenizer that notes the date each prompt was rendered with, read back out of the rendered text."""

    def __init__(self, tok):
        self._tok = tok
        self.seen = []

    def apply_chat_template(self, *args, **kwargs):
        out = self._tok.apply_chat_template(*args, **kwargs)
        found = re.search(r"Today: (\d\d \w{3} \d{4})", out)
        self.seen.append(found.group(1) if found else None)
        return out

    def __call__(self, *args, **kwargs):
        return self._tok(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._tok, name)


def test_a_pinned_date_is_what_a_dated_template_writes_whatever_the_clock_says(clock):
    tok = _real_tokenizer(DATED)
    assert firsttoken.render_chat(tok, "hi") == "Today: 22 Oct 2026|hi>"
    clock.past_midnight()
    assert firsttoken.render_chat(tok, "hi") == "Today: 23 Oct 2026|hi>"
    assert firsttoken.render_chat(tok, "hi", date="2026-10-22") == "Today: 22 Oct 2026|hi>"
    assert firsttoken.render_chat(tok, "hi", date="2025-01-05") == "Today: 05 Jan 2025|hi>"
    assert firsttoken.render_chat(tok, "hi") == "Today: 23 Oct 2026|hi>"      # the pin was let go again


def test_the_pin_is_let_go_when_the_render_fails(clock):
    tok = _real_tokenizer("{{ raise_exception('no') }}")
    with pytest.raises(Exception, match="no"):
        firsttoken.render_chat(tok, "hi", date="2026-10-22")
    from transformers.utils import chat_template_utils
    assert chat_template_utils.datetime is clock


def test_a_template_without_a_date_is_unaffected_by_the_pin(clock):
    tok = _real_tokenizer(UNDATED)
    assert firsttoken.render_chat(tok, "hi") == firsttoken.render_chat(tok, "hi", date="2020-02-02") == "hi>"


def test_a_pin_that_cannot_be_applied_is_refused_not_ignored(monkeypatch):
    from transformers.utils import chat_template_utils
    monkeypatch.delattr(chat_template_utils, "datetime")
    with pytest.raises(RuntimeError, match="cannot pin the template date"), \
            firsttoken.pinned_template_date("2026-10-22"):
        pass


def test_a_malformed_pin_is_refused_by_the_renderer_too():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        firsttoken.render_chat(_real_tokenizer(DATED), "hi", date="today")


def test_only_templates_that_ask_for_the_date_are_called_dated():
    assert chattemplate.uses_date(_real_tokenizer(DATED)) and not chattemplate.uses_date(_real_tokenizer(UNDATED))
    assert chattemplate.uses_date(StubTok()) is False
    assert chattemplate.uses_date(type("T", (), {"chat_template": {"default": UNDATED, "tool_use": DATED}})())


def _pinned_subject(date, log=lambda *_: None):
    spy = Spy(_real_tokenizer(DATED))
    subject = H.HFSubject(_real_model(), spy, batch=3, log=log, template_date=date)
    return subject, spy


PROMPTS = [f"request number {i}{'x' * (i % 4)}" for i in range(7)]


def test_every_path_that_renders_a_prompt_uses_the_pinned_date_across_a_simulated_midnight(clock, monkeypatch):
    """The readout, the lengths, the residuals, the replies, the labels, the stand-in and the generation check."""
    monkeypatch.setattr(H, "REPLY_TOKENS", 4)
    spy = Spy(_real_tokenizer(DATED))
    model = _real_model()
    parts = _parts(8, confirm=True)
    direction = torch.ones(16) / 4

    def fresh():
        # A new subject for each path, so an encoding cached by an earlier path can't hide a render.
        subject = H.HFSubject(model, spy, batch=3, log=lambda *_: None, template_date="2026-10-22")
        subject.refusal_ids, subject.comply_ids = torch.tensor([3, 5]), torch.tensor([7, 9])
        return subject
    paths = {
        "padded_tokens": lambda s: s.padded_tokens(PROMPTS),
        "lengths": lambda s: s._by_length(PROMPTS[::-1]),
        "readout": lambda s: s.readout(PROMPTS, direction, (1,)),
        "front_chain": lambda s: s.readout(PROMPTS, direction, (0, 1)),
        "residuals": lambda s: s.residuals(PROMPTS),
        "replies": lambda s: s.replies(PROMPTS, direction, (1,)),
        "flags": lambda s: s.refusal_flags(PROMPTS, direction, (0,)),
        "labels": lambda s: s._label(PROMPTS),
        "prepare": lambda s: s.prepare(parts),
        "predictive_check": lambda s: H.predictive_check(s, parts, [1], direction, [0, 1, 2], seed=0, conv=FAST),
        "standing_edit": lambda s: (s.edit_standing(direction, (1,)), s.residuals(PROMPTS)),
    }
    for name, call in paths.items():
        before = len(spy.seen)
        clock.past_midnight()
        call(fresh())
        rendered = spy.seen[before:]
        assert rendered, f"{name} rendered nothing, so this test did not reach it"
        assert set(rendered) == {"22 Oct 2026"}, (name, set(rendered))


def test_a_run_across_midnight_gives_the_numbers_of_one_that_never_crossed_it(clock, monkeypatch):
    monkeypatch.setattr(H, "REPLY_TOKENS", 4)
    direction = torch.ones(16) / 4

    def numbers(date, midnight_after):
        FakeClock.today = dt.datetime(2026, 10, 22, 23, 59, 0)  # noqa: DTZ001 - a naive local clock, like the one a template reads
        subject, spy = _pinned_subject(date)
        subject.refusal_ids, subject.comply_ids = torch.tensor([3, 5]), torch.tensor([7, 9])
        out = []
        for step in range(4):
            if step == midnight_after:
                FakeClock.past_midnight()
            out.append(subject.readout(PROMPTS, direction, (step % 2,)))
            out.append(subject.residuals(PROMPTS[:3]))
        return out
    never = numbers("2026-10-22", midnight_after=99)
    across = numbers("2026-10-22", midnight_after=2)
    assert all(torch.equal(a, b) for a, b in zip(never, across, strict=True))


def test_without_the_pin_the_same_run_does_change_across_midnight_which_is_what_the_pin_prevents(clock, monkeypatch):
    direction = torch.ones(16) / 4
    subject, spy = _pinned_subject(None)
    subject.refusal_ids, subject.comply_ids = torch.tensor([3, 5]), torch.tensor([7, 9])
    first = subject.readout(PROMPTS, direction, (1,))
    clock.past_midnight()
    other = H.HFSubject(subject.model, spy, batch=3, log=lambda *_: None)
    other.refusal_ids, other.comply_ids = subject.refusal_ids, subject.comply_ids
    second = other.readout(PROMPTS, direction, (1,))
    assert {"22 Oct 2026", "23 Oct 2026"} <= set(spy.seen)
    assert not torch.equal(first, second)


def test_a_dated_template_with_no_pinned_date_is_warned_about_and_an_undated_one_is_not():
    said = []
    _pinned_subject(None, log=said.append)
    assert len(said) == 1 and "writes today's date" in said[0] and "--template-date" in said[0]
    said.clear()
    _pinned_subject("2026-10-22", log=said.append)
    H.HFSubject(_real_model(), _real_tokenizer(UNDATED), batch=3, log=said.append)
    assert said == []


def test_a_subject_refuses_a_pin_that_is_not_a_date():
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        H.HFSubject(_real_model(), _real_tokenizer(DATED), template_date="22 Oct 2026")


def test_the_run_takes_the_date_once_and_records_it(clock, monkeypatch):
    lines = []
    assert R.choose_template_date("2026-01-02", True, lines.append) == ("2026-01-02", "flag")
    assert R.choose_template_date("2026-01-02", False, lines.append) == ("2026-01-02", "flag")
    assert R.choose_template_date(None, False, lines.append) == (None, None)
    assert lines == []
    monkeypatch.setattr(R, "_today", lambda: "2026-10-22")
    assert R.choose_template_date(None, True, lines.append) == ("2026-10-22", "clock")
    assert "--template-date 2026-10-22" in lines[0]


def _dated_run(monkeypatch, out, **flags):
    """The whole command on a real tiny model whose template writes the date, readable by a label on the first token."""
    from senbonzakura import metrics
    monkeypatch.setattr(H, "REPLY_TOKENS", 3)
    monkeypatch.setattr(metrics, "is_refusal", lambda text: bool(text) and ord(text[0]) % 2 == 0)
    identity = dict(STUB_IDENTITY)
    monkeypatch.setattr(H, "load", lambda *_a, **_k: (_real_model(), _real_tokenizer(DATED), identity))
    monkeypatch.setattr(H, "build_partitions", lambda *_a, **_k: (_parts(confirm=True), {"measure": {"n": 40}}))
    monkeypatch.setattr(A, "selftest", lambda **_k: PASSED)
    args = R.build_parser().parse_args(["stub", "--track", "t", "--out", str(out), "--n-measure", "40", "--batch", "8"])
    for key, value in flags.items():
        setattr(args, key, value)
    return args


def _without_how_it_went(path):
    doc = json.loads((path / "attribution.json").read_text())
    for key in ("created", "run_settings"):
        doc.pop(key)
    doc["sampling"].pop("template_date_source")
    return json.dumps(doc, sort_keys=True)


def test_a_run_records_its_date_and_a_repeat_on_another_day_with_that_date_matches_it(clock, monkeypatch, tmp_path):
    monkeypatch.setattr(R, "_today", lambda: "2026-10-22")
    first, repeat = tmp_path / "first", tmp_path / "repeat"
    R.run(_dated_run(monkeypatch, first), log=lambda *_: None)
    sampling = json.loads((first / "attribution.json").read_text())["sampling"]
    assert sampling["template_dated"] is True and sampling["template_date"] == "2026-10-22"
    assert sampling["template_date_source"] == "clock"
    clock.past_midnight()
    monkeypatch.setattr(R, "_today", lambda: "2026-10-23")
    R.run(_dated_run(monkeypatch, repeat, template_date="2026-10-22"), log=lambda *_: None)
    again = json.loads((repeat / "attribution.json").read_text())["sampling"]
    assert again["template_date"] == "2026-10-22" and again["template_date_source"] == "flag"
    assert _without_how_it_went(first) == _without_how_it_went(repeat)


def test_an_arm_saved_under_another_date_is_not_read_back_as_this_runs(clock, monkeypatch, tmp_path):
    monkeypatch.setattr(R, "_today", lambda: "2026-10-22")
    R.run(_dated_run(monkeypatch, tmp_path), log=lambda *_: None)
    same, other = [], []
    R.run(_dated_run(monkeypatch, tmp_path), log=same.append)
    monkeypatch.setattr(R, "_today", lambda: "2026-10-23")
    R.run(_dated_run(monkeypatch, tmp_path), log=other.append)
    assert any("reusing" in line for line in same)
    assert not any("reusing" in line for line in other)


def test_an_undated_template_records_no_date_and_keeps_its_old_fingerprint(monkeypatch, tmp_path):
    R.run(_stub_run(monkeypatch, tmp_path), log=lambda *_: None)
    sampling = json.loads((tmp_path / "attribution.json").read_text())["sampling"]
    assert sampling["template_dated"] is False and sampling["template_date"] is None
    assert "template_date" not in json.loads((tmp_path / "arm-disjoint.json").read_text())["fingerprint"]


def test_a_follow_up_uses_the_date_the_original_pinned():
    assert F._recorded_template_date({"sampling": {"template_date": "2026-10-22"}}) == "2026-10-22"
    assert F._recorded_template_date({"sampling": {"template_date": None}}) is None
    assert F._recorded_template_date({"sampling": {}}) is None
    assert F._recorded_template_date({}) is None


# ------------------------------------------------------------------------- the fragile flag

def _original_rule(r, c, arm):
    """The membership rule as it was written before it was shared, copied here to hold the new one to it."""
    if r + c < H.MIN_TOKEN_COUNT:
        return None
    if arm == H.ARM_DISJOINT:
        pure_r, pure_c = c == 0, r == 0
    else:
        pure_r, pure_c = r / (r + c) >= H.PURITY_THRESHOLD, c / (r + c) >= H.PURITY_THRESHOLD
    return "refusal" if pure_r else "compliance" if pure_c else None


@pytest.mark.parametrize("arm", [H.ARM_DISJOINT, H.ARM_PURITY])
def test_the_membership_rule_is_unchanged_over_every_small_count(arm):
    for r in range(41):
        for c in range(41):
            assert H._side(r, c, arm) == _original_rule(r, c, arm), (r, c)
    refused, complied = {1: 50, 2: 2, 3: 1, 4: 0, 5: 9}, {1: 0, 2: 0, 3: 3, 4: 40, 5: 1}
    ref, com, dropped = H.token_sets(refused, complied, arm)
    tokens = sorted(set(refused) | set(complied))
    want = {t: _original_rule(refused.get(t, 0), complied.get(t, 0), arm) for t in tokens}
    assert ref == [t for t in tokens if want[t] == "refusal"]
    assert com == [t for t in tokens if want[t] == "compliance"]
    assert dropped == sum(1 for t in tokens if want[t] is None and refused.get(t, 0) + complied.get(t, 0) >= 2)


def _fragile(refused, complied, arm=H.ARM_DISJOINT):
    return {t: (k, now, then) for t, _, _, k, now, then in H.fragile_tokens(refused, complied, arm)}


def test_sets_with_plenty_of_replies_behind_every_token_are_not_flagged():
    assert _fragile({10: 52, 11: 8}, {20: 40, 21: 9}) == {}


def test_a_token_that_only_just_cleared_the_count_threshold_is_flagged_as_resting_on_one_reply():
    assert _fragile({10: 50, 11: 2}, {20: 40}) == {11: (1, "refusal", None)}


def test_a_token_three_replies_deep_rests_on_two():
    assert _fragile({10: 50, 11: 3}, {20: 40}) == {11: (2, "refusal", None)}
    assert _fragile({10: 50, 11: 4}, {20: 40}) == {}


def test_a_token_one_contrary_reply_keeps_out_of_a_set_is_flagged_which_is_the_llama_shape():
    # Three compliant openers and one refusal: without that refusal it would have joined the compliance set.
    assert _fragile({10: 56, 12: 1}, {20: 30, 12: 3}) == {12: (1, None, "compliance")}
    # The same token with eight compliant and none against it is not.
    assert _fragile({10: 52}, {20: 30, 12: 8}) == {}


def test_the_purity_arm_flags_a_token_sitting_on_the_threshold():
    assert _fragile({1: 18}, {1: 2}, H.ARM_PURITY) == {1: (1, "refusal", None)}
    assert _fragile({1: 100}, {1: 1}, H.ARM_PURITY) == {}


def test_a_token_with_one_reply_or_none_can_never_be_made_to_join_by_leaving_replies_out():
    assert _fragile({1: 1}, {2: 1, 3: 0}) == {}


def _fragile_subject(refused_counts, complied_counts, arm=H.ARM_DISJOINT):
    subject = H.HFSubject(StubModel(), StubTok(), batch=8, arm=arm, log=lambda *_: None)
    return subject._fragility(refused_counts, complied_counts)


def test_the_flag_is_recorded_beside_the_token_sets_with_its_rule_and_the_tokens_behind_it():
    flagged = _fragile_subject({10: 50, 11: 2}, {20: 40})
    assert flagged["fragile_readout"] is True and flagged["fragile_readout_token_count"] == 1
    assert flagged["fragile_readout_tokens"] == [{"token": "tok11", "refusal_replies": 2, "compliance_replies": 0,
                                                  "replies_to_change": 1, "now": "refusal", "without_them": None}]
    assert "2 fit replies or fewer" in flagged["fragile_readout_rule"]
    calm = _fragile_subject({10: 50}, {20: 40})
    assert calm["fragile_readout"] is False and calm["fragile_readout_tokens"] == []


def test_the_recorded_list_is_capped_but_the_count_is_not():
    many = _fragile_subject(dict.fromkeys(range(30), 2), {99: 40})
    assert len(many["fragile_readout_tokens"]) == 8 and many["fragile_readout_token_count"] == 30


def test_the_flag_changes_neither_the_token_sets_nor_the_readout(monkeypatch):
    parts = _parts()
    subject = H.HFSubject(StubModel(), StubTok(), batch=8, log=lambda *_: None)
    subject.prepare(parts)
    ids = (subject.refusal_ids.tolist(), subject.comply_ids.tolist())
    sets = subject.agreement_detail["token_sets"]
    assert "fragile_readout" in sets
    refused, complied = subject._fit_counts
    assert H.token_sets(refused, complied, subject.arm)[:2] == (ids[0], ids[1])


def test_the_run_flags_a_fragile_arm_in_the_artefact_the_limits_and_the_log(monkeypatch, tmp_path):
    real = H.HFSubject._fragility

    def fragile(self, refused, complied):
        out = real(self, refused, complied)
        if self.arm == H.ARM_PURITY:
            out.update(fragile_readout=True, fragile_readout_token_count=1,
                       fragile_readout_tokens=[{"token": "tok11", "refusal_replies": 2, "compliance_replies": 0,
                                                "replies_to_change": 1, "now": "refusal", "without_them": None}])
        return out
    monkeypatch.setattr(H.HFSubject, "_fragility", fragile)
    lines = []
    R.run(_stub_run(monkeypatch, tmp_path), log=lines.append)
    doc = json.loads((tmp_path / "attribution.json").read_text())
    assert doc["readout_fragile"] == {"disjoint": False, "purity": True}
    assert any("purity arm: the readout's token sets rest on one or two replies" in x for x in doc["limits"])
    warning = [x for x in lines if x.startswith("warning: the purity arm's readout is fragile")]
    assert len(warning) == 1 and "tok11 (2 refusals, 0 compliant)" in warning[0]
    assert not any("disjoint arm's readout is fragile" in x for x in lines)


def test_a_run_whose_readout_is_not_fragile_says_nothing_and_records_false(monkeypatch, tmp_path):
    lines = []
    R.run(_stub_run(monkeypatch, tmp_path), log=lines.append)
    doc = json.loads((tmp_path / "attribution.json").read_text())
    assert set(doc["readout_fragile"]) == {"disjoint", "purity"}
    assert not any("is fragile" in x for x in lines)


def test_the_spelt_out_number_of_replies_matches_the_constant():
    assert {"one": 1, "two": 2, "three": 3}[R.FRAGILE_REPLIES_TEXT] == H.FRAGILE_REPLIES
