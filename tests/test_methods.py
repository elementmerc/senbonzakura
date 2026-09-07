# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Named recipes, so a comparison is a comparison rather than two runs.

WHY THIS FILE EXISTS

Published abliteration methods differ along three axes and get discussed as though they differ
along one: whether the surgery changes the LENGTH of each weight row or only its DIRECTION,
whether the strength is fixed or SEARCHED, and whether one direction is removed or a SPAN. A
comparative study ranks the optimisation family worst on capability retention and credits the
single-pass family's magnitude/direction split for the difference.

Checked in our own source before any of this was built: `orthogonalize_np_` already captures the
row norms, normalises, subtracts, and restores the norms. The edit primitive the study credits is
already ours, so the axis that actually differs is the search on top. Naming the recipes is what
lets that be measured instead of argued about.

WHAT THE NAMES DELIBERATELY DO NOT SAY

None of these is somebody else's tool. Calling an arm `deccp` would claim we had reproduced their
implementation when what we would have measured is ours.
"""
import pytest

from senbonzakura import methods
from senbonzakura.parser import build_parser


def test_every_method_is_reachable_from_the_command_line():
    accepted = None
    for a in build_parser()._actions:
        if "--method" in a.option_strings:
            accepted = set(a.choices)
    assert accepted == set(methods.CHOICES)


def test_the_default_is_the_searched_recipe():
    """Changing the default would silently change what every existing run means."""
    assert methods.DEFAULT_METHOD == "searched"
    assert build_parser().parse_args(["--model", "m"]).method == "searched"


def test_an_unknown_method_lists_what_exists():
    with pytest.raises(KeyError, match="Available:"):
        methods.get("deccp")


def test_no_recipe_claims_to_be_another_project():
    """`resembles` is a comparison; the NAME must not be an attribution.

    A run named after somebody else's tool would report a statement about their code when it is a
    statement about ours.
    """
    for name in methods.CHOICES:
        low = name.lower()
        for project in ("deccp", "heretic", "erisforge", "orba", "abliterix"):
            assert project not in low, f"{name} names another project"


def test_every_recipe_says_where_it_sits_on_all_three_axes():
    """The axes are the whole point; a recipe that does not place itself on them is a preset."""
    for name in methods.CHOICES:
        m = methods.get(name)
        assert m.edit and m.search and m.directions, f"{name} is missing an axis"
        assert m.summary


def test_the_description_disclaims_reimplementation():
    text = "\n".join(methods.describe("single-pass"))
    assert "NOT a reimplementation" in text
    assert "statement about senbonzakura" in text


# ── the arms have to actually differ ─────────────────────────────────────────────────

def test_the_searched_recipe_pins_no_profile_so_it_searches():
    assert methods.get("searched").settings.get("bake_profile") is None


@pytest.mark.parametrize("name", ["single-pass", "single-pass-raw"])
def test_a_single_pass_recipe_pins_a_bakeable_profile(name):
    """THE POINT OF THE RECIPE. It must produce something the bake path can take directly, or the
    run would search anyway and the arm would be the default wearing another name.
    """
    from senbonzakura.crashsafe import config_to_bake_args

    profile = methods.get(name).settings["bake_profile"]
    bpr, k, mode, _di = config_to_bake_args(profile)
    assert k == 1, "a single pass removes one direction"
    assert mode == "per_layer"
    assert len(bpr) == 8


def test_a_single_pass_profile_is_uniform_across_the_window():
    """Equal max and min weight is what makes the strength flat rather than a searched shape."""
    profile = methods.get("single-pass").settings["bake_profile"]
    for key in ("o_profile", "d_profile"):
        _pos, hi, lo, _dist = profile[key]
        assert hi == lo == 1.0, f"{key} is not a flat full-strength profile"


def test_the_two_single_pass_arms_differ_only_in_the_norm_restoration():
    """They are a controlled pair: same directions, same strength, same layers. If they differed
    in anything else, a difference between them would not be attributable to the edit.
    """
    a = methods.get("single-pass").settings
    b = methods.get("single-pass-raw").settings
    assert a["bake_profile"] == b["bake_profile"]
    assert b.get("no_good_orth") is True
    assert a.get("no_good_orth") is None


def test_the_recipes_are_distinct_settings_not_distinct_labels():
    """Two names for one configuration would produce two arms that cannot differ, and a comparison
    between them would read as evidence that the method does not matter.
    """
    seen = {}
    for name in methods.CHOICES:
        key = repr(sorted(methods.get(name).settings.items()))
        assert key not in seen, f"{name} and {seen[key]} are the same configuration"
        seen[key] = name


# ── the run honours the recipe ───────────────────────────────────────────────────────

def test_a_pinned_recipe_makes_the_run_skip_the_search(monkeypatch):
    """A recipe whose whole claim is "minutes rather than an hour" must not run the search and
    then discard its answer: that would burn the time the recipe exists to save and leave a study
    on disk describing a run that did not use it.
    """
    import types

    from senbonzakura import cli

    obj = cli.Abliterator.__new__(cli.Abliterator)
    obj.args = types.SimpleNamespace(method="single-pass")
    assert obj._method_profile() is not None
    obj.args = types.SimpleNamespace(method="searched")
    assert obj._method_profile() is None


def test_a_namespace_with_no_method_behaves_as_the_default():
    """The forward-only commands share this class's helpers through namespaces that never had the
    flag, and a KeyError there would break paths that do not abliterate at all.
    """
    import types

    from senbonzakura import cli

    obj = cli.Abliterator.__new__(cli.Abliterator)
    obj.args = types.SimpleNamespace()
    assert obj._method_profile() is None
