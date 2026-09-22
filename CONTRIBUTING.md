# Contributing

Thanks for looking. This is a small project with one maintainer, so the process is short.

## Before you write code

Open an issue first if the change is more than a fix. It saves you building something that
does not fit, and it is the fastest way to find out whether the thing you want already
exists under a different name.

## The three things a pull request needs

**1. Tests.** Every new function needs one, and every error path needs one. The suite runs
with `python -m pytest`, and coverage is gated at 95%, so an untested branch fails CI
rather than merging quietly.

**2. A commit message that says why.** What changed is visible in the diff. Why it changed
is not, and in six months it is the only part anyone needs.

**3. A line in `CONTRIBUTORS.md`.** See below.

## The Contributor Licence Agreement

Add one line to `CONTRIBUTORS.md` in your first pull request:

```
Name <email>  YYYY-MM-DD  I agree to the CLA in CLA.md.
```

That is the whole process. No form, no account, no bot.

[`CLA.md`](CLA.md) explains what you are agreeing to and why it is asked for. The short
version: the project is AGPL and stays AGPL, you keep the copyright in what you write, and
the agreement keeps open the option of offering the code under different terms to somebody
who cannot use AGPL code. Doing that later needs permission from every past contributor,
which is often impossible to collect once people have moved on.

If the CLA is a problem for you, say so in the pull request. It is better to discuss it
than to have you walk away.

## What this project is fussy about

Some of these will look excessive for a change of three lines. They exist because each one
is a mistake this project has already made and published.

- **A measurement is not a result until it has an interval.** A single run is a sample.
- **Nothing is fitted and measured on the same rows.** The track builder refuses to write a
  split that leaks, and that refusal is a feature rather than an obstacle.
- **One surface per job.** Three copies of the same prompt-rendering code drifted apart
  here and put the measurement's read-out on the wrong token, which silently changed every
  published number. If you find yourself writing a fourth copy of something, that is the
  bug.
- **Errors are loud.** No caught exception is swallowed, and no failure path is left to
  produce a plausible wrong answer instead of stopping.
- **British English** in prose, comments and user-facing strings.

## Reporting a security issue

Do not open a public issue. Email ops@themalwarefiles.com, and see
[SECURITY.md](SECURITY.md) for what to include and what to expect.

## Reporting a wrong number

If a figure this project publishes does not reproduce, that is the most valuable issue you
can file. Include the command, the version, and what you got. Published numbers have been
corrected here before and will be again.

## Never paste prompts or generations

**Do not put model generations, or the prompts that produced them, into an issue, a pull request,
a discussion or a commit.** That includes the per-prompt rows this tool keeps by default, which
are the most useful debugging artefact here and the last thing anyone wants in a public
repository. They contain harmful prompts and the replies a model gave to them.

Report the shape of the problem instead: the command, the version, the figure you got, and the
figure you expected. If a defect genuinely cannot be described without the text, say so in the
issue and wait to be asked rather than pasting it.

A pre-commit hook refuses a staged artefact carrying a `prompt` or `generation` field, and CI
runs the same check over the whole tree. Install it in one command:

```sh
sh tools/hooks/install-local-hooks.sh
```

It writes a single `pre-commit` hook that runs the tracked gate script,
refuses to overwrite a hook you already have, and is short enough to read first. Until
2026-09-22 this paragraph described a hook no contributor could install, because the hooks it
meant are excluded from the clone.

The check also refuses a fenced block or a blockquote in a results note under
`head-to-head/results/`, because that is the shape a pasted generation arrives in. It is
deliberately not a list of words: a guard carrying a denylist of things a generation might say
would publish those terms in this repository, which is the mistake it exists to prevent.

Two limits worth knowing rather than discovering. **No hook can see a web form**, so on the
issue and discussion path the sentence above is the only control there is. And the shape rule
cannot see a generation retyped as ordinary prose.
