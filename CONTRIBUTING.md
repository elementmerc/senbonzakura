# Contributing

Thanks for looking. This is a small project with one maintainer, so the process is short.

## Before you write code

Open an issue first if the change is more than a small fix. It saves you building something that does not fit. It is also the quickest way to find out whether the thing you want already exists under another name.

## What a pull request needs

1. **Tests.** Every new function needs one, and so does every error path. Run `python -m pytest`. Coverage is gated at 95%, so an untested branch fails CI.
2. **A commit message that says why.** The diff shows what changed. Only the message can say why.
3. **A line in `CONTRIBUTORS.md`.** See below.

## The Contributor Licence Agreement

Add one line to `CONTRIBUTORS.md` in your first pull request:

```
Name <email>  YYYY-MM-DD  I agree to the CLA in CLA.md.
```

No form, no account, no bot. [`CLA.md`](CLA.md) says what you agree to and why. In short, the project is AGPL and stays AGPL, you keep the copyright in what you write, and the agreement keeps open the option of offering the code under other terms. Changing terms later would need permission from every past contributor, which is often impossible once people have moved on.

If the CLA is a problem for you, say so in the pull request.

## What this project is fussy about

Some of these will look excessive for a change of three lines. Each one is a mistake this project has already made and published.

- **A measurement is not a result until it has an interval.** A single run is a sample.
- **Nothing is fitted and measured on the same rows.** The track builder refuses to write a split that leaks, and that refusal is a feature.
- **One surface per job.** Copies of the same logic drift apart. Three copies of the prompt-rendering code put the measurement's read-out on the wrong token, which silently changed every published number. If you find yourself writing a fourth copy, that is the bug.
- **Errors are loud.** No caught exception is swallowed, and no failure path is left to produce a plausible wrong answer.
- **British English** in prose, comments and user-facing strings.

## Reporting a bug or a wrong number

Use the issue templates: [bug report](https://github.com/elementmerc/senbonzakura/issues/new?template=bug-report.md) for a defect, and [wrong number](https://github.com/elementmerc/senbonzakura/issues/new?template=wrong-number.md) for a figure that does not reproduce. Include the command, the version, and what you got.

A wrong number is the most valuable issue you can file. Published numbers have been corrected here before.

## Asking for support

Questions about how to use the tool, or what a number means, go to [Discussions](https://github.com/elementmerc/senbonzakura/discussions). Issues are for defects. [`SUPPORT.md`](SUPPORT.md) has the detail.

## Reporting a security issue

Do not open a public issue. Email ops@themalwarefiles.com, and see [SECURITY.md](SECURITY.md) for what to include and what to expect.

## Never paste prompts or generations

**Do not put model generations, or the prompts that produced them, into an issue, a pull request, a discussion or a commit.** That includes the per-prompt rows this tool keeps by default. They contain harmful prompts and the replies a model gave to them.

Describe the shape of the problem instead: the command, the version, the figure you got, and the figure you expected. If a defect cannot be described without the text, say so and wait to be asked.

A pre-commit hook refuses a staged artefact that carries a `prompt` or `generation` field. CI runs the same check over the whole tree. Install the hook in one command:

```sh
sh tools/hooks/install-local-hooks.sh
```

The installer refuses to overwrite a hook you already have. The check also refuses a fenced block or a blockquote in a results note under `head-to-head/results/`, because that is how a pasted generation usually arrives. It checks the shape of the text, not the words, so it keeps no list of banned terms.

Two limits. No hook can see a web form, so on the issue and discussion path the sentence above is the only control there is. And the shape rule cannot see a generation retyped as ordinary prose.
