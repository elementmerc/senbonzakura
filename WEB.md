# The site moved

The marketing site lived at `web/` in this repository until 2026-10-15. It is now its own
repository and it is **proprietary**, deliberately, while this one stays AGPL-3.0-or-later.

**Why, in one paragraph.** The site is where accounts, sessions and rate limiting will live. Under
a copyleft licence anybody could read exactly how a session is validated, and the commercial
option this project's funding depends on would be closed off on the day it shipped. Nothing about
the tool changes: closing the tool's source would protect nothing, because the method is
publishable and a competent team reimplements it. The decision and its reasoning are Q-121.

It was moved while the site still had no secrets in it, which was checked rather than assumed:
every match for "token" in the tracked files was a CSS design token. Moving it then was a rename;
moving it after session validation landed would have been a migration with a leak in the middle.

**Its history came with it.** The 23 commits that touched `web/` are the real history in the new
repository, not a squashed import, so `git log` and `git blame` still answer there.

**Nothing here depends on it any more.** The workbench bundle build used to take esbuild, react
and react-dom from `web/node_modules`; it now pins them itself in `tools/workbench-ui/`, and the
rebuilt bundle hashes identically to prove the move changed nothing.
