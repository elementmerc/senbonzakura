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
| `UD-Q4_K_XL`, `UD-IQ2_M` | Unsloth Dynamic. A per-model recipe that varies precision layer by layer, not a llama.cpp type | **No.** The recipe is what makes it, and it lives outside the format |
| `i1-Q4_K_M` | Ordinary `Q4_K_M` built with an importance matrix | **Yes**, though not bit for bit: see below |
| `Q4_K_M-GGUF` | A repository naming habit, not a type | Yes, it is just `Q4_K_M` |

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
recording the quantiser build, the type, and the importance matrix if there was one. `senbonzakura
imatrix` writes a `.calibration.json` recording what text the matrix was built from. If you are
comparing two files, read those two sidecars before you read the numbers.

## Where next

- [The track](/guide/the-track) for keeping your evaluation rows honest.
- [Limits](/guide/limits) for every other condition attached to the numbers here.
