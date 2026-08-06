# Install


```sh
pip install .          # or:  uv pip install .   (torch, transformers, accelerate, datasets, optuna)
senbonzakura --help    # console command; equivalently: python -m senbonzakura --help
```

To score a large model on a low-VRAM card, `pip install ".[quant]"` adds 4-bit
(bitsandbytes) loading **for the scorer**: `python -m senbonzakura.score --load-in-4bit`.
The abliterator itself runs in full precision, because it rewrites weights and 4-bit
tensors can't be orthogonalised in place, so `--load-in-4bit` is a measurement option, not
an abliteration one. A man page is installed to `share/man/man1/senbonzakura.1`.

Shell completion for bash, zsh, and tcsh needs `pip install ".[completion]"`, then is
generated on demand (the same one time step `pip`, `gh`, and `poetry` use):
`senbonzakura --print-completion bash | sudo tee /etc/bash_completion.d/senbonzakura`, or the
zsh/tcsh equivalent for your shell. Without that extra the flag is simply absent; nothing
else changes.

### Running the tests

```sh
python -m venv .venv && .venv/bin/pip install --upgrade pip
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU build, no GPU needed
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest                                                     # add --cov for coverage
```

The suite runs entirely on CPU against small hand-built fixtures, so it needs no model
download and no GPU.

If you plan to commit, wire the local gates once per clone:

```sh
bash tools/install-local-hooks.sh
```

That connects `tools/check_prompt_artefacts.py` to your pre-commit path. The tool keeps
per-prompt margins and generations by default, because every scoring bug in this project's
history was invisible in the percentages and obvious in the text, and those rows hold
harmful prompts and the replies a model gave to them. The gate refuses any staged JSON or
JSONL carrying a `prompt` or `generation` field. CI runs the same check over the tree, but
by the time CI sees it the commit exists.

Supported architectures: dense transformers (Llama, Qwen, Mistral, Gemma, Phi and the
like), fused-expert MoE (Qwen3-MoE, Granite-MoE), Mixtral (fused or unfused), OLMoE, and
shared-expert MoE (Qwen2-MoE, DeepSeek-MoE). An unsupported layout fails loudly at load
rather than silently under-ablating.
