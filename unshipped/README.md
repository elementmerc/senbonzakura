# Unshipped

Code that is written, tested and deliberately **not in the wheel yet**.

A module lands here when it is finished enough to keep and not yet reachable from any command.
It stays out of `src/` so that no packaging rule has to remember to exclude it: `setuptools`
discovers packages under `src/`, this directory is not under `src/`, and a file that cannot be
discovered cannot be shipped by accident. A rule in `pyproject.toml` can be deleted in a refactor;
a directory outside the package root cannot.

## Why that matters rather than being tidiness

`rented.py` is the reason this directory exists. It is a RunPod REST client: it reads
`RUNPOD_API_KEY` from the environment and makes HTTPS calls. It was imported by nothing except its
own test, and it shipped in every wheel from the release it was written in.

Shipping an unreachable network client that handles a credential is attack surface bought for no
benefit. Nobody could use it, and everybody installed it.

## Getting something out of here

Move it back into `src/senbonzakura/`, wire it to a command, and give it the tests a shipped
surface gets. The move is the smaller half of that work.

## What is here

- `rented.py` — rented-hardware driver for people with no GPU. See its own docstring for the
  design and for what is deliberately not finished.
