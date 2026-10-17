# Which quantisations this tool can reproduce

Quantisation shrinks a model by storing its numbers less precisely. Names like `Q4_K_M`, `IQ3_XXS`, `i1-Q4_K_M` and
`UD-Q4_K_XL` look like code. This page says which ones `senbonzakura quantise` can produce, which it can only read, and
which aren't llama.cpp quantisations at all. It matters because **a number measured on one quantisation isn't comparable
with a number measured on another**.

In a name, `Q4` is the bits per weight (fewer bits, smaller file, more damage), `_K` is a k-quant that spends its bits
unevenly, `_S` / `_M` / `_L` is the size within the family, `IQ` is a newer scheme for very low bit counts, and `i1-` is a
publisher's prefix for "built with an importance matrix". `i1-` is a convention in filenames, not part of the format.

## What we can produce

`senbonzakura quantise --type` accepts these sixteen, and they are the only ones this tool can make:

```
Q2_K   Q3_K_S  Q3_K_M  Q3_K_L
Q4_K_S Q4_K_M  Q5_K_S  Q5_K_M
Q6_K   Q8_0    Q4_0    Q5_0
IQ4_XS IQ4_NL  F16     BF16
```

Any of them can be built with or without an importance matrix, using `--imatrix`.

## What we can read but not produce

`senbonzakura fetch` reads a file's header and says what it really is. It recognises twenty more types:

```
F32  Q1_0  Q2_K_S  Q4_1  Q5_1
IQ1_S  IQ1_M  IQ2_XXS  IQ2_XS  IQ2_S  IQ2_M
IQ3_XXS  IQ3_XS  IQ3_S  IQ3_M
TQ1_0  TQ2_0  MXFP4_MOE  NVFP4
```

These can be checked, measured and served, but not rebuilt here. **A comparison against one of them is not fully
reproducible here.** If reproducibility matters, requantise both sides into one of the sixteen above.

## Names that are not llama.cpp quantisations

| Name | What it actually is | Can we reproduce it? |
|---|---|---|
| `UD-Q4_K_XL`, `UD-IQ2_M` | Unsloth Dynamic. A per-model recipe that varies precision layer by layer, not a llama.cpp type | **Most of the recipe, never the file.** The layer precisions can be copied, the importance matrix can't (see below) |
| `i1-Q4_K_M` | Ordinary `Q4_K_M` built with an importance matrix | **Yes**, though not bit for bit |
| `Q4_K_M-GGUF` | A repository naming habit, not a type | Yes, it is just `Q4_K_M` |

## Copying a layer-by-layer recipe: `--like`

A `UD-` build has two parts. The precision each tensor got is written in the GGUF header, so it can be read. The
importance matrix that chose those precisions is a calibration run. It isn't in the file, and it is not published.

```
senbonzakura quantise mymodel-BF16.gguf --like Qwen3-0.6B-UD-Q3_K_XL.gguf
```

This applies the reference's per-tensor precisions to your model. It refuses if the two files aren't the same model,
because a recipe aimed at the wrong model lands on whichever layers share names.

Your output therefore has their layer plan and your own matrix, or none. That's why it's never named `UD-` anything. It's
called `mymodel-Q3_K_M-copied-schedule.gguf`, and the run prints a `NOT COPIED:` line for the missing half. **That missing
half decides whether a score measured on their file carries over to yours.** The `.provenance.json` beside the output
records the reference's name and sha256, the copied precisions, and `"importance_matrix_copied": false`.

A real run of `Qwen3-0.6B-BF16.gguf` against Unsloth's `Qwen3-0.6B-UD-Q3_K_XL.gguf` printed:

```
schedule: 290 of 310 tensors carry the reference's type, 20 differ.
```

The twenty are known before the run starts. They are stored as `IQ3_S` and `IQ3_XXS`, which `llama-quantize` won't take
as a per-tensor instruction, so they fall back to the base recipe. The count is printed because `llama-quantize` will
accept an instruction, ignore it, and still produce a file. **Read the count before any number measured on the file.**

## The one that catches people out

Two files can both say `Q4_K_M` and be measurably different models, because one was built with an importance matrix and
the other wasn't. An importance matrix records which weights mattered most on some calibration text, and **the text
changes the result**. Two `i1-Q4_K_M` files from different publishers aren't necessarily the same build. One comparison
was once run with one side `i1-Q4_K_M` and the other plain `Q4_K_M`. Someone caught it by reading the filenames.

So `senbonzakura quantise` writes a `.provenance.json` beside every file. It records the quantiser build, the type, the
importance matrix if any, and the sha256 of the input and output. `senbonzakura imatrix` writes a `.calibration.json`
with the text the matrix came from. Read both before comparing two files.

To build the matrix in the same command:

```
senbonzakura quantise model.gguf out.gguf --type Q4_K_M \
    --imatrix auto --imatrix-file prose.txt
```

`--imatrix auto` makes one, and `--imatrix FILE` uses one you already have. The tool won't pick the calibration text for
you, because the text is part of the result. Broad, ordinary prose is the usual choice. The corpora bundled here are
refusal-measurement sets, and calibrating on them before measuring refusal puts the thing being measured into the
instrument.

## Where next

- [The track](/guide/the-track), for keeping your evaluation rows honest.
- [Limits](/guide/limits), for the other conditions attached to the numbers here.
