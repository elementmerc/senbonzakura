# Which models it can open

Checked before anything downloads: `senbonzakura doctor` reports every architecture
module that imports on your install.


- Dense transformers: Llama, Qwen, Mistral, Gemma, Phi and the rest of that shape.
- Fused-expert mixture-of-experts: Qwen3-MoE, Granite-MoE.
- Mixtral, fused or unfused.
- OLMoE.
- Shared-expert MoE: Qwen2-MoE, DeepSeek-MoE.
- LFM2, including its MoE variant. These are **hybrids**: some of their layers hold a short
  convolution where other models hold attention, and that convolution writes into the model's
  running state exactly as attention does. On LFM2.5-350M it is 10 layers out of 16. Those are
  edited too, because editing the other six and reporting success would be an abliteration that
  never reached most of the model.

::: tip New words: dense and mixture-of-experts
A **dense** model runs every one of its weights on every token. A **mixture-of-experts**
model keeps a pile of specialist sub-networks and routes each token to a couple of them, so
it's big on disk and cheap to run. It matters here because the two store their weights in
different shapes, and abliteration is weight surgery: you have to know which drawer things
are in.
:::

::: warning Quantised uploads can't be abliterated, and the popular ones are quantised
Abliteration is weight surgery: it rewrites real matrices in place. A 4-bit or 8-bit upload
doesn't store those matrices in a form that can be rewritten, so the tool refuses it rather than
pretending. That includes the GGUF files most local runners use, and it includes the
`bnb-4bit` uploads that repackage popular models at half the size.

This catches people out because those uploads have a well-earned reputation for being smaller at
no cost to quality, so they're the natural thing to reach for. Start from the original
full-precision repository instead. You can quantise afterwards; you can't abliterate a
quantisation.
:::

An architecture it doesn't recognise **fails loudly at load** with the layer type named. It
would be easy to make it shrug and carry on, and the result would be a model that came back
looking abliterated and wasn't, because the edit never reached the layers that mattered.
That isn't a hypothetical. It's precisely what happened on Gemma for months, and it cost
this project every Gemma number it had ever published.


## Where next

- [Install](/guide/install) if you have not got it yet.
- [The method](/guide/how-it-works) for why the shape of the model matters at all.
