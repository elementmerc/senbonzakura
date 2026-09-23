# Bundled corpora

These prompt sets ship inside the wheel so the tool runs with no network. Each is redistributed under its own licence, and MIT and CC-BY both require the notice to travel with the work. This file is generated from the same table the loader reads, so it cannot fall out of step with what actually ships.

```
AdvBench harmful behaviours  [advbench]
    Zou et al. 2023, Universal and Transferable Adversarial Attacks on Aligned Language Models
    MIT · llm-attacks/llm-attacks @ a62d1307e38b · data/advbench/harmful_behaviors.csv
    520 prompts, used as the harmful arm

HarmBench standard behaviours  [harmbench]
    Mazeika et al. 2024, HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal
    MIT · centerforaisafety/HarmBench @ c0423b952435 · data/behavior_datasets/harmbench_behaviors_text_all.csv
    200 prompts, used as the harmful arm

HarmBench copyright behaviours  [harmbench-copyright]
    Mazeika et al. 2024, HarmBench
    MIT · centerforaisafety/HarmBench @ c0423b952435 · data/behavior_datasets/harmbench_behaviors_text_all.csv
    100 prompts, used as the harmful arm

StrongREJECT forbidden prompts  [strongreject]
    Souly et al. 2024, A StrongREJECT for Empty Jailbreaks
    MIT · alexandrasouly/strongreject @ 3432b2d696b4 · strongreject_dataset/strongreject_dataset.csv
    313 prompts, used as the harmful arm

XSTest safe prompts (over-refusal)  [xstest-safe]
    Röttger et al. 2024, XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours
    CC-BY-4.0 · paul-rottger/exaggerated-safety @ 475f10bf0a3d · xstest_prompts.csv
    250 prompts, used as the benign arm

XSTest unsafe prompts  [xstest-unsafe]
    Röttger et al. 2024, XSTest
    CC-BY-4.0 · paul-rottger/exaggerated-safety @ 475f10bf0a3d · xstest_prompts.csv
    200 prompts, used as the harmful arm
```

## The bundled capability probe

Separate from the corpora above, and listed separately because it is a different kind of thing.
The corpora are prompt sets used to find and score refusal. This is a graded benchmark used to
measure what the edit COST: the model either gets each question right or it does not.

It ships inside the package and the capability gate uses it by default, so a run measures
capability without needing a download or a dataset card.

```
GSM8K, grade-school arithmetic  [capability-gsm8k]
    Cobbe et al. 2021, Training Verifiers to Solve Math Word Problems
    MIT · openai/gsm8k @ 740312add88f · main::test
    256 of the split's 1,319 rows, question and reference answer, used as the capability probe
```

The 256 are selected by ordering the split on a digest of each question and taking the first 256,
which is reproducible from the pinned revision alone and does not depend on a random number
generator or on the order the rows arrived in. `tools/packaging/build_capability_probe.py` rebuilds
the file and can check the committed copy against what the pin rebuilds to.

Unlike the corpora, this file is committed rather than generated. The corpora are generated because
they hold harmful prompts and are published as a gated dataset on purpose; none of that applies to
arithmetic, and generating it would leave a clone install unable to run a gate that is on by
default.
