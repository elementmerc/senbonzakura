# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
"""Tests for the dataset resolver (senbonzakura/dataset.py).

The resolver exists because the tool used to accept exactly one input shape, and the failure on
the commonest shape in the ecosystem was actively misleading rather than merely unhelpful: a
DatasetDict reported "holds 1 prompt" (counting splits) and then claimed there was no `text`
column, which was false. So a good share of these tests are about the errors, not the successes.
"""
import json

import pytest
from datasets import Dataset, DatasetDict

from senbonzakura import dataset
from senbonzakura import dataset as ds


# ── parse_spec ───────────────────────────────────────────────────────────────────
def test_a_bare_spec_has_no_split_or_slice():
    assert ds.parse_spec("mytrack/bad_ds") == ("mytrack/bad_ds", None, None)


def test_a_split_is_parsed():
    assert ds.parse_spec("owner/name::train") == ("owner/name", "train", None)


def test_a_split_and_slice_are_parsed():
    assert ds.parse_spec("owner/name::train[:400]") == ("owner/name", "train", ":400")


def test_a_windows_drive_colon_is_not_a_split():
    # A single colon cannot be told apart from a drive letter, which is why the separator is `::`.
    assert ds.parse_spec(r"C:\corpus\bad_ds") == (r"C:\corpus\bad_ds", None, None)


def test_an_empty_spec_says_so():
    with pytest.raises(ds.DatasetError, match="no dataset was given"):
        ds.parse_spec("   ")


# ── hub id detection ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("body", ["mlabonne/harmful_behaviors", "walledai/AdvBench"])
def test_hub_ids_are_recognised(body):
    assert ds.looks_like_hub_id(body)


@pytest.mark.parametrize("body", ["./local/dir", "/abs/path", "~/corpus", "file.csv", "plain"])
def test_paths_are_not_hub_ids(body):
    assert not ds.looks_like_hub_id(body)


# ── column selection ─────────────────────────────────────────────────────────────
def test_text_wins_when_present():
    assert ds.pick_column(["goal", "text"], None, "s") == "text"


@pytest.mark.parametrize("col", ["prompt", "instruction", "goal", "behavior", "question"])
def test_the_well_known_corpora_columns_are_detected(col):
    # AdvBench uses `goal`, HarmBench `behavior`, Alpaca `instruction`. Each of these previously
    # needed a conversion pass before the tool would look at the file.
    assert ds.pick_column([col, "other"], None, "s") == col


def test_a_named_column_beats_detection():
    assert ds.pick_column(["text", "goal"], "goal", "s") == "goal"


def test_a_named_column_that_is_absent_lists_what_there_is():
    with pytest.raises(ds.DatasetError) as e:
        ds.pick_column(["a", "b"], "missing", "s")
    assert "'a', 'b'" in str(e.value).replace('"', "'")
    assert "--text-column" in str(e.value)


def test_no_recognisable_column_names_everything_it_tried():
    with pytest.raises(ds.DatasetError) as e:
        ds.pick_column(["x", "y"], None, "s")
    assert "--text-column" in str(e.value)
    assert "goal" in str(e.value)


def test_a_chat_column_is_detected_when_nothing_plainer_exists():
    assert ds.pick_column(["messages"], None, "s") == "messages"


# ── chat flattening ──────────────────────────────────────────────────────────────
def test_a_plain_string_passes_through():
    assert ds.flatten_chat("hello", "s") == "hello"


def test_the_last_user_turn_is_the_prompt():
    turns = [{"role": "system", "content": "be nice"},
             {"role": "user", "content": "first"},
             {"role": "assistant", "content": "reply"},
             {"role": "user", "content": "second"}]
    assert ds.flatten_chat(turns, "s") == "second"


def test_the_assistant_reply_is_never_the_prompt():
    # Fitting a refusal direction on text the MODEL produced rather than on the request would be
    # measuring the wrong thing entirely.
    turns = [{"role": "user", "content": "ask"}, {"role": "assistant", "content": "answer"}]
    assert ds.flatten_chat(turns, "s") == "ask"


def test_the_sharegpt_from_value_shape_works_too():
    turns = [{"from": "human", "value": "ask"}, {"from": "gpt", "value": "answer"}]
    assert ds.flatten_chat(turns, "s") == "ask"


def test_a_row_with_no_user_turn_is_refused():
    with pytest.raises(ds.DatasetError, match="no user turn"):
        ds.flatten_chat([{"role": "assistant", "content": "only me"}], "s")


def test_a_chat_cell_of_the_wrong_type_is_refused():
    with pytest.raises(ds.DatasetError, match="expected a list"):
        ds.flatten_chat(42, "s")


# ── slicing ──────────────────────────────────────────────────────────────────────
def test_a_prefix_slice():
    assert ds._apply_slice(list(range(10)), ":4", "s") == [0, 1, 2, 3]


def test_a_suffix_slice():
    assert ds._apply_slice(list(range(10)), "8:", "s") == [8, 9]


def test_a_bounded_slice():
    assert ds._apply_slice(list(range(10)), "2:4", "s") == [2, 3]


def test_a_bare_count_slice():
    assert ds._apply_slice(list(range(10)), "3", "s") == [0, 1, 2]


def test_a_nonsense_slice_says_what_a_good_one_looks_like():
    with pytest.raises(ds.DatasetError, match=r"\[:400\]"):
        ds._apply_slice([1], "a:b", "s")


def test_a_three_part_slice_is_refused():
    with pytest.raises(ds.DatasetError):
        ds._apply_slice([1], "1:2:3", "s")


# ── the DatasetDict defect ───────────────────────────────────────────────────────
def test_a_single_split_datasetdict_needs_no_argument(tmp_path):
    p = tmp_path / "dd"
    DatasetDict({"train": Dataset.from_dict({"text": ["a", "b"]})}).save_to_disk(str(p))
    assert ds.resolve(str(p)) == ["a", "b"]


def test_a_multi_split_datasetdict_names_its_splits_instead_of_lying(tmp_path):
    """The regression test for the defect. The old reader said "holds 2 prompts" and then
    "no 'text' column", and both statements were false.
    """
    p = tmp_path / "dd"
    DatasetDict({"train": Dataset.from_dict({"text": ["a"]}),
                 "test": Dataset.from_dict({"text": ["b"]})}).save_to_disk(str(p))
    with pytest.raises(ds.DatasetError) as e:
        ds.resolve(str(p))
    msg = str(e.value)
    assert "splits" in msg
    assert "train" in msg and "test" in msg
    assert "no 'text' column" not in msg


def test_a_named_split_of_a_datasetdict_is_read(tmp_path):
    p = tmp_path / "dd"
    DatasetDict({"train": Dataset.from_dict({"text": ["a"]}),
                 "test": Dataset.from_dict({"text": ["b"]})}).save_to_disk(str(p))
    assert ds.resolve(f"{p}::test") == ["b"]


def test_an_unknown_split_lists_the_real_ones(tmp_path):
    p = tmp_path / "dd"
    DatasetDict({"train": Dataset.from_dict({"text": ["a"]})}).save_to_disk(str(p))
    with pytest.raises(ds.DatasetError, match="no split named 'valid'"):
        ds.resolve(f"{p}::valid")


# ── the original shape still works ───────────────────────────────────────────────
def test_a_plain_save_to_disk_directory_still_works(tmp_path):
    p = tmp_path / "bad_ds"
    Dataset.from_dict({"text": ["a", "b", "c"]}).save_to_disk(str(p))
    assert ds.resolve(str(p)) == ["a", "b", "c"]


def test_a_save_to_disk_directory_with_another_column(tmp_path):
    p = tmp_path / "adv"
    Dataset.from_dict({"goal": ["a", "b"]}).save_to_disk(str(p))
    assert ds.resolve(str(p)) == ["a", "b"]


def test_an_empty_dataset_says_it_is_empty(tmp_path):
    p = tmp_path / "e"
    Dataset.from_dict({"text": []}).save_to_disk(str(p))
    with pytest.raises(ds.DatasetError, match="empty"):
        ds.resolve(str(p))


def test_a_missing_path_that_is_not_a_hub_id_says_both_things():
    with pytest.raises(ds.DatasetError) as e:
        ds.resolve("nowhere-at-all")
    assert "does not look like a Hub dataset id" in str(e.value)


# ── files ────────────────────────────────────────────────────────────────────────
def test_a_plain_text_file_is_one_prompt_per_line(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("one\ntwo\n\nthree\n", encoding="utf-8")
    assert ds.resolve(str(p)) == ["one", "two", "three"]


def test_a_csv_detects_its_column(tmp_path):
    p = tmp_path / "advbench.csv"
    p.write_text("goal,target\nmake a bomb,Sure\nhack a car,Sure\n", encoding="utf-8")
    assert ds.resolve(str(p)) == ["make a bomb", "hack a car"]


def test_a_tsv_works(tmp_path):
    p = tmp_path / "p.tsv"
    p.write_text("text\tlabel\nalpha\tx\nbeta\ty\n", encoding="utf-8")
    assert ds.resolve(str(p)) == ["alpha", "beta"]


def test_a_jsonl_file(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text('{"prompt": "a"}\n\n{"prompt": "b"}\n', encoding="utf-8")
    assert ds.resolve(str(p)) == ["a", "b"]


def test_a_broken_jsonl_line_names_the_line(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text('{"prompt": "a"}\n{oops\n', encoding="utf-8")
    with pytest.raises(ds.DatasetError, match="line 2"):
        ds.resolve(str(p))


def test_a_json_list(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps([{"text": "a"}, {"text": "b"}]), encoding="utf-8")
    assert ds.resolve(str(p)) == ["a", "b"]


def test_a_json_list_of_bare_strings(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    assert ds.resolve(str(p)) == ["a", "b"]


def test_a_json_object_with_a_rows_key(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"data": [{"text": "a"}]}), encoding="utf-8")
    assert ds.resolve(str(p)) == ["a"]


def test_a_json_object_with_no_rows_key_says_what_it_wanted(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"nope": 1}), encoding="utf-8")
    with pytest.raises(ds.DatasetError, match="list of rows"):
        ds.resolve(str(p))


def test_a_parquet_file(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa
    p = tmp_path / "p.parquet"
    pq.write_table(pa.table({"instruction": ["a", "b"]}), str(p))
    assert ds.resolve(str(p)) == ["a", "b"]


def test_a_slice_applies_to_a_file_the_same_way(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("\n".join(str(i) for i in range(10)) + "\n", encoding="utf-8")
    assert ds.resolve(f"{p}::train[:3]") == ["0", "1", "2"]


def test_a_slice_that_selects_nothing_says_so(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("a\nb\n", encoding="utf-8")
    with pytest.raises(ds.DatasetError, match="selected no rows"):
        ds.resolve(f"{p}::train[5:]")


def test_limit_bounds_the_read(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("\n".join("abcde") + "\n", encoding="utf-8")
    assert ds.resolve(str(p), limit=2) == ["a", "b"]


def test_a_file_of_only_blank_lines_is_empty(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("\n\n\n", encoding="utf-8")
    with pytest.raises(ds.DatasetError, match=r"empty|blank"):
        ds.resolve(str(p))


# ── the hub path, without touching the network ───────────────────────────────────
def _fake_datasets(monkeypatch, loader):
    """Patch `load_dataset` on the real module rather than replacing the module.

    Swapping `sys.modules["datasets"]` wholesale sends `Dataset` operations into infinite
    recursion, because the library re-imports itself internally. `from datasets import
    load_dataset` inside the resolver resolves the attribute at call time, so patching the
    attribute is enough and leaves everything else working.
    """
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", loader, raising=False)


def test_a_hub_id_is_fetched_with_its_split(monkeypatch):
    seen = {}

    def loader(body, **kw):
        seen.update(body=body, **kw)
        return Dataset.from_dict({"text": ["a", "b"]})

    _fake_datasets(monkeypatch, loader)
    assert ds.resolve("owner/name::train[:1]", token=None) == ["a"]
    assert seen["body"] == "owner/name"
    assert seen["split"] == "train"


def test_a_token_is_passed_through(monkeypatch):
    seen = {}

    def loader(body, **kw):
        seen.update(kw)
        return Dataset.from_dict({"text": ["a"]})

    _fake_datasets(monkeypatch, loader)
    ds.resolve("owner/name", token="secret")  # noqa: S106 - a stub loader, not a credential
    assert seen["token"] == "secret"


def test_the_environment_token_is_used_when_none_is_passed(monkeypatch):
    seen = {}

    def loader(body, **kw):
        seen.update(kw)
        return Dataset.from_dict({"text": ["a"]})

    _fake_datasets(monkeypatch, loader)
    monkeypatch.setenv("HF_TOKEN", "from-env")
    ds.resolve("owner/name")
    assert seen["token"] == "from-env"


def test_a_gated_dataset_says_how_to_get_in(monkeypatch):
    def loader(body, **kw):
        raise RuntimeError("401 Client Error: gated repo")

    _fake_datasets(monkeypatch, loader)
    with pytest.raises(ds.DatasetError) as e:
        ds.resolve("owner/name")
    assert "--hf-token" in str(e.value)


def test_an_ordinary_hub_failure_does_not_claim_it_is_gated(monkeypatch):
    def loader(body, **kw):
        raise RuntimeError("connection reset")

    _fake_datasets(monkeypatch, loader)
    with pytest.raises(ds.DatasetError) as e:
        ds.resolve("owner/name")
    assert "gated" not in str(e.value)


def test_streaming_stops_at_the_limit(monkeypatch):
    def loader(body, **kw):
        assert kw.get("streaming") is True

        def gen():
            for i in range(10_000):
                yield {"text": f"row{i}"}
        return gen()

    _fake_datasets(monkeypatch, loader)
    out = ds.resolve("owner/name::train", streaming=True, limit=3)
    assert out == ["row0", "row1", "row2"]


# ── resolve_labelled ─────────────────────────────────────────────────────────────
def test_a_labelled_csv_splits_into_two_sides(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("text,label\na,harmful\nb,harmless\nc,harmful\n", encoding="utf-8")
    harmful, harmless = ds.resolve_labelled(str(p))
    assert harmful == ["a", "c"]
    assert harmless == ["b"]


def test_a_labelled_file_accepts_a_named_label_value(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("text,category\na,unsafe\nb,safe\n", encoding="utf-8")
    harmful, harmless = ds.resolve_labelled(str(p), harmful_values=("unsafe",))
    assert harmful == ["a"] and harmless == ["b"]


def test_a_labelled_file_with_no_matching_label_says_which_flag_fixes_it(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("text,label\na,red\nb,blue\n", encoding="utf-8")
    with pytest.raises(ds.DatasetError, match="--harmful-label"):
        ds.resolve_labelled(str(p))


def test_a_labelled_file_that_is_all_harmful_is_refused(tmp_path):
    # A refusal rate with no harmless arm cannot tell abliteration from brain damage.
    p = tmp_path / "c.csv"
    p.write_text("text,label\na,harmful\nb,harmful\n", encoding="utf-8")
    with pytest.raises(ds.DatasetError, match="no harmless prompts"):
        ds.resolve_labelled(str(p))


def test_a_labelled_file_with_no_label_column_says_so(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("text,other\na,1\n", encoding="utf-8")
    with pytest.raises(ds.DatasetError, match="--label-column"):
        ds.resolve_labelled(str(p))


def test_a_plain_text_file_cannot_be_split_by_label(tmp_path):
    p = tmp_path / "p.txt"
    p.write_text("a\nb\n", encoding="utf-8")
    with pytest.raises(ds.DatasetError, match="no labels in it"):
        ds.resolve_labelled(str(p))


def test_labelled_chat_rows_work(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(
        json.dumps({"messages": [{"role": "user", "content": "a"}], "label": "harmful"}) + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "b"}], "label": "harmless"}) + "\n",
        encoding="utf-8")
    harmful, harmless = ds.resolve_labelled(str(p))
    assert harmful == ["a"] and harmless == ["b"]


# ── the paths coverage would otherwise leave untested ────────────────────────────
def test_an_empty_slice_expression_changes_nothing():
    assert ds._apply_slice([1, 2], "  ", "s") == [1, 2]


def test_a_dict_chat_cell_is_treated_as_one_turn():
    assert ds.flatten_chat({"role": "user", "content": "solo"}, "s") == "solo"


def test_non_dict_turns_are_skipped_rather_than_crashing():
    assert ds.flatten_chat(["junk", {"role": "user", "content": "real"}], "s") == "real"


def test_a_turn_with_no_content_is_skipped():
    turns = [{"role": "user"}, {"role": "user", "content": "real"}]
    assert ds.flatten_chat(turns, "s") == "real"


def test_strip_false_keeps_padding_and_blank_rows(tmp_path):
    """The bench path depends on this. Heretic strips on read and senbonzakura does not, so a
    padded prompt is a real difference between the tools and the slice builder must still see it.
    """
    p = tmp_path / "p.txt"
    p.write_text("  padded  \n\nplain\n", encoding="utf-8")
    assert ds.resolve(str(p), strip=False) == ["  padded  ", "", "plain"]


def test_a_json_of_the_wrong_type_is_refused(tmp_path):
    p = tmp_path / "p.json"
    p.write_text("42", encoding="utf-8")
    with pytest.raises(ds.DatasetError, match="expected a list"):
        ds.resolve(str(p))


def test_broken_json_names_the_file(tmp_path):
    p = tmp_path / "p.json"
    p.write_text("{oops", encoding="utf-8")
    with pytest.raises(ds.DatasetError, match="not valid JSON"):
        ds.resolve(str(p))


def test_an_unreadable_directory_names_the_role(tmp_path):
    (tmp_path / "notadataset").mkdir()
    with pytest.raises(ds.DatasetError, match="could not load the contrast set"):
        ds.resolve(str(tmp_path / "notadataset"), what="contrast set")


def test_every_row_blank_in_the_chosen_column_says_which_column(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("text,other\n  ,1\n  ,2\n", encoding="utf-8")
    with pytest.raises(ds.DatasetError, match="blank in column 'text'"):
        ds.resolve(str(p))


def test_labelled_reads_a_save_to_disk_directory(tmp_path):
    p = tmp_path / "d"
    Dataset.from_dict({"text": ["a", "b"], "label": ["harmful", "harmless"]}).save_to_disk(str(p))
    harmful, harmless = ds.resolve_labelled(str(p))
    assert harmful == ["a"] and harmless == ["b"]


def test_labelled_refuses_a_named_label_column_that_is_absent(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("text,label\na,harmful\nb,x\n", encoding="utf-8")
    with pytest.raises(ds.DatasetError, match="no column named 'nope'"):
        ds.resolve_labelled(str(p), label_column="nope")


def test_labelled_refuses_a_path_that_does_not_exist():
    with pytest.raises(ds.DatasetError, match="not a Hub id"):
        ds.resolve_labelled("nowhere-at-all-either")


def test_labelled_reads_a_hub_id(monkeypatch):
    def loader(body, **kw):
        return Dataset.from_dict({"text": ["a", "b"], "label": ["harmful", "harmless"]})

    _fake_datasets(monkeypatch, loader)
    harmful, harmless = ds.resolve_labelled("owner/name")
    assert harmful == ["a"] and harmless == ["b"]


def test_labelled_applies_a_slice_to_both_sides(tmp_path):
    p = tmp_path / "c.csv"
    rows = ["text,label"] + [f"h{i},harmful" for i in range(5)] + [f"g{i},harmless" for i in range(5)]
    p.write_text("\n".join(rows) + "\n", encoding="utf-8")
    harmful, harmless = ds.resolve_labelled(f"{p}::train[:2]")
    assert harmful == ["h0", "h1"] and harmless == ["g0", "g1"]


def test_an_empty_labelled_source_says_so(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("text,label\n", encoding="utf-8")
    with pytest.raises(ds.DatasetError, match="empty"):
        ds.resolve_labelled(str(p))


# ── the dataset CONFIG, which had no way to be expressed ───────────────────────────
# Found by running E2 on 2026-09-07: `openai/gsm8k` holds two configs and refuses to load without
# one being named, `load_dataset` takes the config as its second POSITIONAL argument, and
# `owner/name::split` carried only a body and a split. The default capability benchmark in this
# project's own experiment script could not be loaded by any invocation of it.
@pytest.mark.parametrize(("spec", "want"), [
    ("openai/gsm8k:main", ("openai/gsm8k", "main")),
    ("owner/name:cfg", ("owner/name", "cfg")),
    ("openai/gsm8k", ("openai/gsm8k", None)),
])
def test_a_hub_config_is_split_off_the_body(spec, want):
    assert dataset.split_hub_config(spec) == want


@pytest.mark.parametrize("spec", [
    r"C:\corpus",              # a Windows drive letter is not a Hub owner
    "./local.csv",
    "/abs/path",
    "plainname",
    "owner/name:",             # a trailing colon names no config
])
def test_a_colon_that_is_not_a_config_is_left_alone(spec):
    assert dataset.split_hub_config(spec) == (spec, None)


def test_a_spec_carrying_a_config_is_still_recognised_as_a_hub_id():
    """Without this it fell through to the path branch and was reported as a spelling mistake."""
    assert dataset.looks_like_hub_id("openai/gsm8k:main")
    assert dataset.looks_like_hub_id("openai/gsm8k")
    assert not dataset.looks_like_hub_id(r"C:\corpus")


def test_the_config_reaches_load_dataset_as_a_positional(monkeypatch):
    """It is the SECOND POSITIONAL argument, which is why it needed its own place in the spec
    rather than another keyword.
    """
    seen = {}

    class _Fake:
        column_names = ("question", "answer")

        def __iter__(self):
            return iter([{"question": "q", "answer": "a"}])

    def fake_load_dataset(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return _Fake()

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    dataset._from_hub("openai/gsm8k:main", "test", "openai/gsm8k:main::test", None, False, None)
    assert seen["args"] == ("openai/gsm8k", "main")
    assert seen["kwargs"]["split"] == "test"


def test_a_missing_config_error_says_how_to_supply_one(monkeypatch):
    """The upstream error lists the configs and never says where to put one."""
    def fake_load_dataset(*_a, **_k):
        raise ValueError("Config name is missing. Please pick one among the available configs: "
                         "['main', 'socratic']")

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    with pytest.raises(dataset.DatasetError) as e:
        dataset._from_hub("openai/gsm8k", "test", "openai/gsm8k::test", None, False, None)
    msg = str(e.value)
    assert "openai/gsm8k:<config>::test" in msg, msg
