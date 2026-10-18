# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
# Author:  Daniel Iwugo
# Comment: Christ is King  # noqa: ERA001
"""A refusal whose cause is a missing earlier step says which step.

THE CLASS THIS COVERS. Some commands sit downstream of another one: `quantise` needs a GGUF that
`convert` produces. Each of them refused correctly and stopped at naming the problem, which leaves
a reader who has not met the pipeline with a true sentence and nothing to do about it.

Under the standing assumption that nobody reads the source, "the thing you need does not exist"
is only half a message. The other half is the command that makes it exist.
"""
import pytest

from senbonzakura import quantise


def test_quantise_points_at_convert_rather_than_only_saying_the_file_is_absent(tmp_path):
    """Still true after `quantise` learned to take a checkpoint itself.

    It was tempting to delete this when the command stopped needing a separate conversion step,
    and the first version of that change did exactly that: the refusal became "it does not exist"
    and named nothing. But somebody reaching this message has a path that is neither a GGUF nor a
    checkpoint, and `convert` is still what makes the first of those. Both routes belong in the
    message: pass the checkpoint here, or make the GGUF yourself if you want it kept.
    """
    with pytest.raises(SystemExit) as caught:
        quantise.main([str(tmp_path / "nothing.gguf"), str(tmp_path / "out.gguf")])
    message = str(caught.value)
    assert "senbonzakura convert" in message, message
    assert "checkpoint" in message, (
        "the refusal names the two-step route and not the one-step one this command now has")


