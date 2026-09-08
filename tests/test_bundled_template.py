# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""A chat template this tool ships, by NAME, for models that carry none.

WHY A NAME AND NOT A FALLBACK

The old behaviour invented a "User:/Assistant:" wrapper whenever a model shipped no template. It
was chosen silently and recorded nowhere, and prompt format drives the refusal rate, the KL and the
compass together, so every number measured that way is comparable to nothing. That is why a missing
template became a refusal.

The refusal is correct and it is also a wall. Base models are a legitimate thing to abliterate and
several ship no template at all; Bamba is the one that forced this, when `validate --experiment
reach` refused it and the only way past was for somebody to write a template by hand, with every
person writing a different one.

So the format is bundled and named. Naming it makes it an input the operator chose rather than
something the tool assumed, puts its digest in the artefact, and makes two runs under the same name
comparable to each other. What has not changed is that the tool still picks nothing on its own.
"""
from __future__ import annotations

import pytest

from senbonzakura import cli

transformers = pytest.importorskip("transformers")


@pytest.fixture
def bare_tok():
    """A tokenizer with no chat template, which is the situation under test."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.chat_template = None
    return tok


def test_a_model_with_no_template_is_still_refused_by_default(bare_tok):
    """THE GUARANTEE THAT MUST SURVIVE. Bundling a format must not make it the default one."""
    with pytest.raises(SystemExit) as e:
        cli.ensure_chat_template(bare_tok)
    msg = str(e.value)
    assert "ships no chat template" in msg
    assert "will not do is pick a format for you" in msg


def test_the_refusal_names_the_bundled_options(bare_tok):
    """A wall with no door is how somebody ends up writing their own template badly."""
    with pytest.raises(SystemExit) as e:
        cli.ensure_chat_template(bare_tok)
    for name in cli.BUNDLED_TEMPLATES:
        assert name in str(e.value)


@pytest.mark.parametrize("name", cli.BUNDLED_TEMPLATES)
def test_every_bundled_template_actually_renders(name, bare_tok):
    """A template that ships and does not work is worse than none, because the refusal that would
    have caught it has already been satisfied.
    """
    prov = cli.ensure_chat_template(bare_tok, name)
    assert prov["sha256"]
    rendered = cli.render_chat(bare_tok, "hello")
    assert "hello" in rendered
    assert rendered.strip(), "an empty prompt would score every model identically"


def test_a_bundled_template_is_recorded_by_name_not_by_path(bare_tok):
    """An absolute path inside somebody's virtualenv differs per install and tells a reader
    nothing. Two runs both saying `bundled:plain` are comparable and can be seen to be.
    """
    prov = cli.ensure_chat_template(bare_tok, "plain")
    assert prov["source"] == "bundled:plain"
    assert "/" not in prov["source"] and "\\" not in prov["source"]


def test_the_same_name_gives_the_same_digest_every_time(bare_tok):
    from transformers import AutoTokenizer

    second = AutoTokenizer.from_pretrained("gpt2")
    second.chat_template = None
    assert (cli.ensure_chat_template(bare_tok, "plain")["sha256"]
            == cli.ensure_chat_template(second, "plain")["sha256"])


def test_an_unknown_name_says_what_the_known_ones_are(bare_tok):
    with pytest.raises(SystemExit) as e:
        cli.ensure_chat_template(bare_tok, "nosuchtemplate")
    msg = str(e.value)
    assert "nosuchtemplate" in msg
    for name in cli.BUNDLED_TEMPLATES:
        assert name in msg


def test_a_path_still_beats_a_name(tmp_path, bare_tok):
    """A file called `plain` in the working directory must not shadow the bundled one, and a real
    path must keep working: the bundled names are a small closed set and a path is anything else.
    """
    p = tmp_path / "mine.jinja"
    p.write_text("{% for m in messages %}<<{{ m['content'] }}>>{% endfor %}"
                 "{% if add_generation_prompt %}!{% endif %}", encoding="utf-8")
    prov = cli.ensure_chat_template(bare_tok, str(p))
    assert prov["source"] == str(p)
    assert "<<hello>>" in cli.render_chat(bare_tok, "hello")


def test_the_bundled_file_is_declared_as_package_data():
    """It is read at runtime, so a wheel without it installs, imports, and raises the moment a
    template-less model is loaded, which is the worst shape of bug: it only appears for users.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    declared = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "data/templates/*.jinja" in declared
    for name in cli.BUNDLED_TEMPLATES:
        assert cli._bundled_template(name).is_file(), f"{name} is named and not shipped"


def test_a_name_that_is_not_bundled_resolves_to_no_bundled_path():
    assert cli._bundled_template("plain") is not None
    assert cli._bundled_template("../../../etc/passwd") is None
    assert cli._bundled_template("") is None


def test_validate_accepts_and_forwards_a_chat_template():
    """`reach` is a `validate` sub-command and `validate` had no such flag, so both blocked reach
    runs died on `unrecognized arguments` naming a flag the operator had just been told to use.

    Forwarding matters as much as accepting: a flag taken here and dropped before the loader sees
    it would start the run, refuse the model for having no template, and name that same flag.
    """
    from senbonzakura import validate

    own, args = validate.build_args(
        ["--model", "m", "--out", "o", "--experiment", "reach", "--chat-template", "plain"])
    assert own.chat_template == "plain"
    assert args.chat_template == "plain", "accepted here and dropped before the loader"


def test_validate_forwards_trust_remote_code():
    from senbonzakura import validate

    _own, args = validate.build_args(
        ["--model", "m", "--out", "o", "--trust-remote-code"])
    assert args.trust_remote_code is True


def test_validate_without_a_template_forwards_nothing():
    """The default must stay empty, or every model gets a template it did not ask for."""
    from senbonzakura import validate

    _own, args = validate.build_args(["--model", "m", "--out", "o"])
    assert args.chat_template == ""
