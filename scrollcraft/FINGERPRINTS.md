# Fingerprints

Every site you build with **scroll-craft** gets one row here, appended after it
ships. The registry exists so your next build can prove it is a different page
rather than a re-skin of one you already made.

This file is **yours**. It starts empty on purpose: the gate is about not
repeating *yourself*, so it has nothing to say until you have built something.

The rules and the gate live in the skill's
`references/uniqueness.md`. Short version:

**A new build must differ from EVERY row below on at least 4 of the 6
dimensions.** Four against each row individually, not four on average across the
table. If a planned build fails, change the plan. Never edit a row to make room
for it.

The six dimensions are: **grammar**, **nav treatment**, **hero device**,
**act-sequence shape**, **close pattern**, **signature move**.

Dimension 6 is free, because a signature move is unique by definition. So the
gate really asks for three more out of the remaining five, and a build that
changes only grammar and world will fail it.

---

## The registry

| Build | Grammar | Nav treatment | Hero device | Act-sequence shape | Close pattern | Signature move | World | Port |
|---|---|---|---|---|---|---|---|---|
| senbon-dev (2026-10-10) | Instrument readout (new; see the build brief) | Sticky bar that is a page surface rather than a brand plate, flush at rest and an inset island with a blur once the page moves. Carries the corrections log as its one opinionated link. | Two masked headline lines rising once on load, then stillness, then a pointer-tracked wash and a magnetic pill | kinetic, colour sweep, authored silence, seed assembly (peak), flow-in stillness, type, ground. 6 acts, 5.77 viewport-heights at 1440x900 and 8.98 at 390x844 | The ground steps hard to the deeper surface and holds. No spotlight, no magnet, no fade out. The last screen has the full link plate and the copyright line on it | **Seed assembly**: the bar chart builds itself out of the five real seed runs each bar is the mean of, five hairlines landing at their own measured heights before the bar is drawn from them, and it retracts on the way back up | Dense editorial. No photography, no footage, no generated assets. Geometry, type and real numbers only | Next.js 16 static export, React 19, anime.js v4. The scroll-craft engine is deliberately not adopted |

*(one row. From the second build onwards, this table is the constraint.)*

---

## What is taken

Add a bullet here whenever a build claims something a later build should avoid
reusing: a grammar, a nav treatment, a close pattern, a signature move, an
act-count-and-length band. The shared columns are what the next build inherits
as a constraint, so writing them down is the whole point.

- **Grammar: instrument readout.** A section is a reading: a claim in prose beside
  the figure it came from, and the figure re-derives itself from its own source
  data as the reader arrives. It forbids pinning, `scrub`, crossfading grounds,
  re-hiding prose, and any figure that animates to a number it was not computed
  from. A later build that wants a prose-beside-evidence page needs a different
  organising logic, not this one with a new palette.
- **Hero device: masked lines on load, then stillness and a pointer response.**
  The restraint is the hero. A later build cannot claim this by swapping the
  wash for a spotlight.
- **Close pattern: a hard ground step that holds.** No spotlight on the last
  stage, no magnetic call to action, no fade.
- **Signature move: seed assembly.** A summary statistic drawing itself out of
  the individual measurements behind it. Taken.
- **Act count and length band: 6 acts at 5.8 viewport-heights on desktop.** Short
  on purpose, because the grammar is editorial. A later short build needs a
  different reason for being short.
- **Not taken, and worth saying so:** no `scrub` act exists here, because this
  build had no footage and no asset generation. The device is still free.

---

## Appending a row

After shipping, add one line to the table and one bullet to **What is taken** if
the build claimed something new. Fill every column. Say what the build shares
with existing rows.

Rows are append-only. A build that has been superseded stays in the table,
because the space it occupies is still occupied.

---

## Worked example

The skill's author kept a registry of twelve builds across eight page grammars.
If you want to see what a filled-in table looks like, and which shapes tend to
collide, read `EXAMPLES.md` in the scroll-craft repository. Treat it as
illustration only: those rows are somebody else's builds and they do **not**
constrain yours.
