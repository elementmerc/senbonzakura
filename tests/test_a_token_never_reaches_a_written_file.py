# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A Hub token given on the command line must never reach a file the run writes.

THE DEFECT, REPRODUCED BEFORE IT WAS FIXED. A run given `--base-licence` writes a model card,
`README.md`, beside the weights, and the card's "Reproducing it" section held `" ".join(sys.argv)`
verbatim. So `--hf-token <value>` went into the one file in the output directory that exists to be
uploaded with the weights. Reproduced end to end on 2026-10-03 with a fake value: the full token
was on line 57 of the card. It shipped in 0.4.0 and 0.4.1. A redactor for exactly this flag
already existed in `measure.py` and this path never called it.

WHY THE ASSERTION IS ON THE VALUE AND NOT ON THE FUNCTION. A test that checks one function strips
one flag proves that function and nothing else, and the card was the second place this project
let a secret through after `measure` logged a raw argv for months. So the end to end tests here
drive a whole run and then read EVERY byte of every file it left behind, looking for the value
itself. A new writer that leaks the token through a path nobody thought of fails here without
anyone having to remember to add it.

The value is assembled at runtime so no token-shaped literal sits in the source, where a secret
scanner would rightly flag it.
"""
import sys
from pathlib import Path

import pytest

from senbonzakura import cli, modelcard

FAKE = "hf_" + "senbonzakuraTESTnotAtoken" + "0" * 10


def _every_file_containing(root, needle):
    """Every file under `root` whose bytes contain `needle`, however the file is encoded."""
    raw = needle.encode("utf-8")
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob("*")
                  if p.is_file() and raw in p.read_bytes())


@pytest.mark.parametrize("spelling", [["--hf-token", FAKE], [f"--hf-token={FAKE}"]],
                         ids=["separate", "equals"])
def test_a_run_writes_the_token_into_no_file_at_all(
        spelling, base_args, tiny_model, tiny_tok, track, tmp_path, monkeypatch):
    """Both spellings argparse accepts, through a whole run, with the card switched on."""
    monkeypatch.setattr(sys, "argv", ["senbonzakura", "kageyoshi", "--model", "tiny",
                                      *spelling, "--base-licence", "apache-2.0",
                                      "--out", str(base_args.out)])
    base_args.hf_token = FAKE
    base_args.base_licence = "apache-2.0"
    base_args.json_events = str(tmp_path / "events.jsonl")
    lines = []
    cli.Abliterator(base_args, lines.append, model=tiny_model, tok=tiny_tok).run()

    card = Path(base_args.out) / "README.md"
    # NOT VACUOUS. A run that wrote nothing would pass a search for the token in nothing.
    assert card.is_file(), "\n".join(lines[-5:])
    assert (Path(base_args.out) / "abliteration.json").is_file()
    assert _every_file_containing(tmp_path, FAKE) == [], (
        "the token reached a file the run wrote")
    assert not any(FAKE in line for line in lines), "the token reached the run's log"
    # The card still says a token was supplied, which a reader reproducing the run needs to know.
    body = card.read_text(encoding="utf-8")
    assert "--hf-token" in body and modelcard.REDACTED in body


def test_the_token_in_the_environment_is_kept_out_of_the_card_too(
        base_args, tiny_model, tiny_tok, track, tmp_path, monkeypatch):
    """The documented route is `$HF_TOKEN`, and a shell expands it into argv if it is passed
    as `--hf-token "$HF_TOKEN"`, or anywhere else a user pastes it. The value is matched as well
    as the flag, so it is caught wherever on the line it lands.
    """
    monkeypatch.setenv("HF_TOKEN", FAKE)
    monkeypatch.setattr(sys, "argv", ["senbonzakura", "kageyoshi", "--model",
                                      f"https://example.invalid/{FAKE}/model",
                                      "--base-licence", "apache-2.0"])
    base_args.base_licence = "apache-2.0"
    cli.Abliterator(base_args, [].append, model=tiny_model, tok=tiny_tok).run()
    assert (Path(base_args.out) / "README.md").is_file()
    assert _every_file_containing(tmp_path, FAKE) == []


@pytest.mark.parametrize("command", [
    f"senbonzakura kageyoshi --model m --hf-token {FAKE} --trials 2",
    f"senbonzakura kageyoshi --model m --hf-token={FAKE}",
    f"senbonzakura kageyoshi --model m --hf-token '{FAKE}'",
    f'senbonzakura kageyoshi --model m --hf-token "{FAKE}"',
    f"senbonzakura kageyoshi --model m --hf-token\t{FAKE}",
], ids=["separate", "equals", "single-quoted", "double-quoted", "tab"])
def test_the_card_redacts_the_flag_whoever_supplies_the_command(command):
    """At the sink, not at one caller. `senbonzakura report --command` takes the command as text
    a person pasted, and nothing upstream of `build` can be relied on to have cleaned it.
    """
    body = "\n".join(modelcard.build({"model": "m"}, None, command, licence="apache-2.0"))
    assert FAKE not in body
    assert "--hf-token" in body, "the reader should still see that a token was used"
    assert "--trials 2" in body or "--trials" not in command, "the rest of the line survives"


def test_a_known_secret_is_redacted_wherever_it_appears_on_the_line():
    body = "\n".join(modelcard.build({"model": "m"}, None, f"run --model {FAKE}",
                                     licence="apache-2.0", secrets=(FAKE,)))
    assert FAKE not in body


def test_an_empty_secret_redacts_nothing():
    """`str.replace("", ...)` inserts between every character, which would wreck the command
    line while looking like caution.
    """
    body = modelcard.build({"model": "m"}, None, "run --model m", licence="apache-2.0",
                           secrets=("", None))
    assert "run --model m" in body


def test_report_writes_no_token_into_the_card_it_saves(tmp_path):
    abl = tmp_path / "abliteration.json"
    abl.write_text('{"model": "m"}', encoding="utf-8")
    out = tmp_path / "README.md"
    assert modelcard.main(["--abliteration", str(abl), "--base-licence", "apache-2.0",
                           "--command", f"senbonzakura kageyoshi --hf-token {FAKE}",
                           "--out", str(out)]) == 0
    assert out.is_file()
    assert _every_file_containing(tmp_path, FAKE) == []
