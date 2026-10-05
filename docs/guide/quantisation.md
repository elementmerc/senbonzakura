# Which quantisations this tool can reproduce

Quantisation shrinks a model by storing its numbers less precisely. A 4 GB model becomes 1 GB and
runs on a laptop, at some cost in quality.

The names look like a code: `Q4_K_M`, `IQ3_XXS`, `i1-Q4_K_M`, `UD-Q4_K_XL`. This page says which
of them `senbonzakura quantise` can actually produce, which it can only read, and which are not
llama.cpp quantisations at all. It matters because **a number measured on one quantisation is not
comparable to a number measured on another**, and the names do not always make the difference
obvious.

## Reading the name

A GGUF filename usually carries three separate pieces of information glued together.

```
        i1-Q4_K_M
        │  │ │ │
        │  │ │ └── size within the family: S small, M medium, L large
        │  │ └──── K-quant family
        │  └────── 4 bits per weight
        └───────── built with an importance matrix
```

- **`Q4`** is how many bits each weight gets. Fewer bits, smaller file, more damage.
- **`_K`** means a k-quant, which spends its bits unevenly and is better than the plain `Q4_0`.
- **`_S` / `_M` / `_L`** pick a size within that family.
- **`IQ`** instead of `Q` is a different, newer scheme that squeezes harder at very low bit counts.
- **`i1-`** is a prefix some publishers use to say an importance matrix was used. It is a
  convention in filenames, not part of the format.

## What we can produce

`senbonzakura quantise --type` accepts these sixteen, and they are the only ones this tool can
make:

```
Q2_K   Q3_K_S  Q3_K_M  Q3_K_L
Q4_K_S Q4_K_M  Q5_K_S  Q5_K_M
Q6_K   Q8_0    Q4_0    Q5_0
IQ4_XS IQ4_NL  F16     BF16
```

Any of them can be built with or without an importance matrix, by passing `--imatrix`. That is the
same choice the `i1-` prefix records.

## What we can read but not produce

`senbonzakura fetch` inspects a downloaded file's header and tells you what it really is, and it
recognises twenty more:

```
F32  Q1_0  Q2_K_S  Q4_1  Q5_1
IQ1_S  IQ1_M  IQ2_XXS  IQ2_XS  IQ2_S  IQ2_M
IQ3_XXS  IQ3_XS  IQ3_S  IQ3_M
TQ1_0  TQ2_0  MXFP4_MOE  NVFP4
```

So a file of one of these types can be checked, measured and served. It cannot be rebuilt from
source by this tool, which means **a comparison against one of them is not fully reproducible
here**. If reproducibility matters for what you are doing, requantise both sides yourself into one
of the sixteen above.

## Names that are not llama.cpp quantisations

Some popular filenames look like quantisation types and are not.

| Name | What it actually is | Can we reproduce it? |
|---|---|---|
| `UD-Q4_K_XL`, `UD-IQ2_M` | Unsloth Dynamic. A per-model recipe that varies precision layer by layer, not a llama.cpp type | **Most of the recipe, never the file.** We can copy the layer-by-layer precisions, because the file records them; on a real `UD-Q3_K_XL` that was 290 of 310 tensors. We cannot copy the importance matrix, because it is not published: see below |
| `i1-Q4_K_M` | Ordinary `Q4_K_M` built with an importance matrix | **Yes**, though not bit for bit: see below |
| `Q4_K_M-GGUF` | A repository naming habit, not a type | Yes, it is just `Q4_K_M` |

## Copying a layer-by-layer recipe: `--like`

A `UD-` build is two things glued together, and only one of them is a secret.

```
        a UD- build
        ├── which precision each tensor got  ──  written in the file. Readable.
        └── the importance matrix that chose  ──  a calibration run. Not in the
            them                                 file, and not published.
```

**The first half is in the file.** A GGUF stores a precision for every tensor in its own header, so
you can read the exact recipe out of any published build without being told it. Think of a shop
selling a cake: you cannot get the recipe, but you can weigh each layer of the one you bought.

```
senbonzakura quantise mymodel-BF16.gguf --like Qwen3-0.6B-UD-Q3_K_XL.gguf
```

That reads the reference's per-tensor precisions and applies the same ones to your model. The run
refuses if the two files are not the same model, because a recipe aimed at the wrong model lands on
the layers whose names happen to match and quietly leaves the rest alone.

**The second half is not in the file.** An importance matrix is a recording made by running the
model over some text (see the section below). It changes which weights keep their precision, it
leaves no trace in the file it produced, and Unsloth does not publish the text they use. So your
output has their layer plan and your own matrix, or none.

That is why the output is never named `UD-` anything. It is called
`mymodel-Q3_K_M-copied-schedule.gguf`, and the run prints a `NOT COPIED:` line saying which half is
missing, because **that missing half is exactly what decides whether a score measured on their file
carries over to yours**. The `.provenance.json` beside the output records the reference's name and
sha256, the precisions that were copied, and a plain `"importance_matrix_copied": false`.

### What it actually gets you

After the run, the output is read back and compared tensor by tensor against the reference, and the
count is printed. Here is a real run of `Qwen3-0.6B-BF16.gguf` against Unsloth's own
`Qwen3-0.6B-UD-Q3_K_XL.gguf`:

```
schedule: 290 of 310 tensors carry the reference's type, 20 differ.
```

**The twenty are told to you before the run starts, not discovered afterwards.** They are the
tensors Unsloth stored as `IQ3_S` and `IQ3_XXS`, and `llama-quantize` will not take those as a
per-tensor instruction, so they fall back to the base recipe. The run says so up front:

```
20 of 310 tensors in the reference carry a type llama-quantize will not accept as a
per-tensor override, so they take the base recipe instead and the output is NOT this
reference's schedule on those tensors (IQ3_S, IQ3_XXS; for example blk.3.attn_k.weight).
```

The other 290 came out at exactly the reference's precision, and the finished file was 357.4 MB
against the reference's 356.6 MB.

The count is printed rather than hidden because `llama-quantize` will accept an instruction, ignore
it, and produce a file anyway. The count is the only evidence either way, so **read it before you
read any number measured on the file**. Two things, in order: how much of the plan landed, and the
fact that the matrix never does.

## The one that catches people out

Two files can both say `Q4_K_M` and be measurably different models, because one was built with an
importance matrix and the other was not. That is what the `i1-` prefix is for, and not everyone
uses it.

An importance matrix is a recording of which weights mattered most on some calibration text. The
quantiser then spends more of its bit budget on those. **Which text was used changes the result**,
so even two `i1-Q4_K_M` files from different publishers are not the same build.

This is not hypothetical. A comparison next door was once run with the abliterated side quantised
`i1-Q4_K_M` and the stock side quantised plain `Q4_K_M`, which is two variables in a one-variable
experiment. It was caught by someone reading the filenames.

Because of that, `senbonzakura quantise` writes a `.provenance.json` beside every file it makes,
recording the quantiser build, the type, the importance matrix if there was one, and the sha256 of
both the file it read and the file it wrote. `senbonzakura imatrix` writes a `.calibration.json`
recording what text the matrix was built from. If you are comparing two files, read those two
sidecars before you read the numbers.

The two hashes are what let you check that the file in your hand is the file the receipt describes,
which is the question a receipt exists to answer. They matter most when the source is gone:
`--prune-source` deletes the halfway file once the output verifies, and without a recorded hash the
pairing cannot be reconstructed afterwards. They are always recorded, never behind a flag, because a
field that appears only when somebody remembered to ask for it is a field nothing downstream can
rely on.

## Where next

- [The track](/guide/the-track) for keeping your evaluation rows honest.
- [Limits](/guide/limits) for every other condition attached to the numbers here.
