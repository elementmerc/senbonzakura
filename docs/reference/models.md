# Which models it can open

Checked before anything downloads: `senbonzakura doctor` lists the architecture modules that import on your install.

- Dense transformers: Llama, Qwen, Mistral, Gemma, Phi and the rest of that shape.
- Fused-expert mixture-of-experts: Qwen3-MoE, Granite-MoE.
- Mixtral, fused or unfused.
- OLMoE.
- Shared-expert MoE: Qwen2-MoE, DeepSeek-MoE.
- LFM2, including its MoE variant. These are **hybrids**: some layers hold a short convolution where other models hold attention. The convolution writes into the running state just as attention does, so those layers are edited too. On LFM2.5-350M that is 10 layers out of 16.

::: tip New words: dense and mixture-of-experts
A **dense** model runs every weight on every token. A **mixture-of-experts** model keeps many specialist sub-networks and routes each token to a few of them. It is big on disk and cheap to run. The two store their weights in different shapes, and abliteration is weight surgery, so the tool needs to know which shape it has.
:::

::: warning Quantised uploads cannot be abliterated
Abliteration rewrites real matrices in place. A 4-bit or 8-bit upload does not store them in a form that can be rewritten, so the tool refuses it. That includes GGUF files and `bnb-4bit` uploads.

Start from the original full-precision repository. You can quantise afterwards. You cannot abliterate a quantisation.
:::

An architecture the tool does not recognise **fails loudly at load**, and names the layer type. A tool that shrugged and carried on would return a model that looks abliterated but was never reached in the layers that matter.

## Where next

- [Install](/guide/install) if you have not got it yet.
- [The method](/guide/how-it-works), for why the shape of the model matters at all.
