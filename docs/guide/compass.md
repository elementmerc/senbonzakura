# The compass

## Checking whether the directions are refusal directions

A method that finds several directions has not thereby shown that the extra ones remove
refusal. They might be removing subject matter. A set of harmful prompts about one topic
differs from harmless prompts partly because of the topic, and cutting that costs the
model knowledge rather than caution.

`senbonzakura validate` is the check. It is separate from the abliterator on purpose:
the tool that produces a direction set should not be the only thing that grades it.

```sh
senbonzakura validate --model <hf-id> --track mytrack --experiment all --out result.json
```

Three questions, and each has a control that can fail:

- **Do the directions work on prompts they were never fitted on?** The harmful prompts are
  grouped, one group is held out, directions are fitted on the rest, and the held-out group
  is scored. A direction tied to a subject cannot separate a subject it never saw. The
  score is reported next to a **random direction** measured the same way, because a number
  without a floor beside it cannot be read.
- **Do the fitted extra directions beat random ones?** The same run repeats with the extra
  directions replaced by random ones. If cutting random directions does as well, the
  fitted ones were not carrying anything.
- **Is removing several better than removing one?** Direction count is swept against
  ablation strength, and the results are compared **at matched refusal removal**. This
  matters more than it sounds: cutting harder always costs more coherence, so comparing
  coherence between runs that removed different amounts of refusal compares nothing.

The run reports an unreadable grid as unreadable. If every setting lands on the same
refusal rate, there is no room for direction count to show an effect, and it says so
instead of printing a table that looks like data.

**We publish our own results from this, including when they are unflattering.** See the
caveat under the table above.
