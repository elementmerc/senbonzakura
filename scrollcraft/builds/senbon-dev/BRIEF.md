# BRIEF: senbon.dev scroll motion

**Self-authored under explicit creative delegation.** The operator delegated
creative direction for the motion rework and supplied four specific asks (counting
figures both ways, the grid colouring both ways, a drift chart that reads without
a legend, a looping terminal). Those are recorded verbatim below as evidence.
Everything else on this page is an authored decision, and it is labelled as one.

This is a **motion rework of an existing page**, not a new build. All copy,
structure, sections and branding are fixed. What is in scope is the grammar of
movement: which device carries which beat, what the peak is, and how the page
paces.

---

## The eight topics

### 1. Vibe, in three to five words, plus references

*Authored.* Laboratory notebook, not a launch. Quiet, exact, slightly austere,
and willing to print its own bad result. References, none of them websites: the
methods section of a decent paper, a calibration certificate that ships with an
instrument, the plates in an old physics textbook where the error bars are drawn
bigger than the effect.

### 2. The scroll journey, section by section

*Fixed by the existing page, not authored here.* Six blocks, and the order is
already the argument:

```
1  Hero       what this is, and the one promise a hosted tool cannot make
2  The gap    other people's numbers for the problem, none of them ours
3  Measured   our own numbers, and the headline one is our flagship losing
4  Honesty    where the long form lives, and every figure's own file
5  Install    the one command
6  Footer
```

`/corrections` is a second route and keeps its own descending-spine treatment.

### 3. The energy curve

*Authored.* Calm open, a rise through other people's numbers, the loudest
moment in the middle on our own loss, then down. The page must not be loud at
the open, because the open is a claim about restraint and a loud hero
contradicts it.

### 4. Feeling curve, and the one moment they remember

**The feeling curve.** One line per act: the emotion, then what on screen causes
it.

```
1  Hero       Composure   two headline lines rise out of masks and stop. Nothing
                          else moves until the reader moves their pointer.
2  The gap    Alarm       175 of 400 squares colour up pink under the reader's
                          own hand, and four figures count while they do it.
3  (silence)  Suspense    an empty run of about three quarters of a screen
                          between the citations and the peak's first word.
                          AUTHORED SILENCE, not dead scroll. See below.
4  Measured   Reversal    "It lost." The chart assembles itself out of the five
                          individual runs it averaged, and one unlucky run is
                          visibly dragging our own number up.
5  Honesty    Relief      nothing moves. The prose arrives once and stays. The
                          only still act on the page, and it is placed where the
                          reader needs to breathe.
6  Install    Readiness   a terminal types the command and its output, and loops.
7  Footer     Resolve     the ground steps a shade darker and holds. The page
                          lands somewhere and stops.
```

No two adjacent acts carry the same feeling. Alarm needs the composure in front
of it, reversal needs the alarm, and relief only works because the peak was loud.

**The peak.** The sentence a visitor would say to a friend:

> it's the site where their own bar chart builds itself out of the five separate
> runs behind it, and you can watch one unlucky run drag their own headline
> number up

It lives in **act 4, Measured**. The proposed peak was the "It lost." act, and
that is kept; what changed is the *mechanism*. The copy says the flagship feature
lost. The motion makes the reader watch *how* it lost, from the real per-seed
numbers, rather than asking them to take the summary on trust.

### 5. One thing this site should do that no site has

*Authored.* **Show its own working while it draws.** Every other measurement
tool's landing page draws a summary. This one assembles the summary out of the
individual measurements in front of the reader, including the outlier, and the
outlier is the reason the summary is unflattering to us.

### 6. How far from premium-minimal

*Fixed.* The page is already built and the palette is a brand kit. Family:
**dense / editorial**, not premium-minimal. One accent (the sakura pink), spent
only on measurements.

### 7. One unbroken world, or distinct scenes?

*Authored.* Distinct scenes with hard grounds. This is a document with
instruments in it, not a place. A continuous world would be a lie about what the
page is, and there is no footage to fly through.

### 8. What assets exist

*Fixed, and the constraint that shapes everything.* No video, no photography, no
generated assets, and **no asset generation is permitted on this build**. There
is no `KIE_AI_API_KEY` and none is to be set. So `scrub` is unavailable, and
every device has to come out of type, geometry, real numbers and the pointer.

---

## The grammar

**A new grammar: `instrument readout`.** The eight defined grammars were each
read against this page and all eight lost. Reasons, in order:

| Grammar | Why it lost |
|---|---|
| Filmic one-shot | Leans on `scrub`, and there is no footage. Also forbids visible sequence, which this page's whole argument depends on. |
| Chaptered editorial | Closest fit, but it forbids a fixed bar and the page has a sticky nav that is not being removed. |
| Live surface | Forbids marketing chrome, a wordmark-plus-CTA bar, and display headings. The page has all three and they are fixed. |
| Continuous world | Requires worldflight and a single canvas. No assets, and the page is a document. |
| Typographic poster | Forbids the charts, and the charts are the product. |
| Gallery / catalog | There is no range of objects. One tool, one result. |
| Split stage | Tempting, because three sections are already prose-beside-figure. But it forbids the hero and requires the divider to be the chrome, and the collapse ending. All three are structural changes the brief forbids. |
| Rhythmic cutlist | Bans `pin`, `dwell` and `parallax`, which is fine, but demands 12 to 20 short cuts. This page is five long prose blocks. |

**What `instrument readout` is.** A section is a **reading**: a claim in prose
beside the figure it came from, and the figure re-derives itself from its own
source data as the reader arrives at it.

- **Navigation:** a sticky bar that is a surface of the page rather than a brand
  plate, and that lifts into an island once the page moves. It carries the one
  link a sceptical reader wants, which is the corrections log.
- **Sequence:** cumulative and visible in the argument, never numbered.
- **Ending:** the page hands over a command, then steps its ground a shade
  darker and holds.
- **Forbids:** pinning of any kind (a page whose subject is other people's
  patience must not take the scroll off the reader); `scrub`; any figure that
  animates to a number it was not computed from; crossfading grounds (each
  section's ground is painted and hard); and re-hiding prose.

That last ban is how this build resolves the tension between the operator's ask
and the skill's rule. scroll-craft says content that re-hides on scroll up is a
defect. The operator asked for figures that count down again on the way out.
Both are honoured by splitting them:

- **Figures and graphics scrub both ways.** They are illustrations of a
  measurement, and a measurement that re-derives itself when you look again is
  the page's whole point.
- **Prose reveals once and stays.** Reading must never re-hide. Before this
  rework every paragraph on the page re-hid, measured at opacity 0 on the way
  back to the top.

The data table in the peak act counts as prose for this purpose, because it is a
table somebody reads, so it wipes in once and stays.

---

## The signature move

**Seed assembly.** The drift chart does not fade in. Each bar is built in front
of the reader out of the five individual seed runs it is the mean of: five
hairline ticks land at their real measured heights, in seed order, and only then
does the bar grow to the mean of them. Scroll back and the ticks lift off and
the bar retracts.

Every height is a real number from
`evidence/k-sweep-2026-08-13/drift-per-seed.json`, which is committed in the
repository and already cited on the page. The ten values are:

```
one direction   0.0468  0.0232  0.0608  0.0700  0.0477   mean 0.0497
two directions  0.0622  0.0719  0.0708  0.0828  0.1784   mean 0.0932
```

The move earns its place because it is the honest reading rather than the
flattering one: seed 46's 0.1784 is more than double the next-worst run in its
own arm, and it is most of why our own headline ratio is 1.9. A reader watching
the bar assemble sees the spread column (0.0482) happen rather than being told
about it.

It is not a kit device, not a parameter change to one, and not something the
engine does.

---

## The tell-someone sentence

> It's the site where the bar chart builds itself out of the five separate runs
> behind it, and you watch one unlucky run drag their own headline number up.

---

## Authored silence

One run of empty scroll is deliberate and must not be read as dead scroll by any
verification pass: the gap between the end of The gap's citations and the first
word of Measured. It is about three quarters of a viewport at 1440x900. It is
the quiet the peak arrives out of, and without it the peak follows a loud act
directly and has nothing to be a change from.

---

## The score table

| Beat | Section | The act's reason (device family) | Support | Scrubs both ways? |
|---|---|---|---|---|
| 1 Composure | Hero | `kinetic` lines, two mask-rises on load | `pointer`: a pointer-tracked wash behind the type, a magnet on the pill. Ambient petal drift. | No, once |
| 2 Alarm | The gap | `colour sweep`: 175 pink cells colour up over a grey base, waved in reading order | `count` on the four figures; `flow+in` once on the prose | Yes, grid and figures |
| 3 Reversal, PEAK | Measured | **`seed assembly`**, the signature move | `reveal` wipe on the table rows, once; `count` on 1.9; a drawn 95% interval with its marker; `grow` on the bars | Yes, except the table |
| 4 Relief | Honesty | `flow+in` and otherwise nothing. The still act. | none, on purpose | No, once |
| 5 Readiness | Install | `type`: the looping transcript | `flow+in` once on the prose | No, it loops |
| 6 Resolve | Footer | `ground`: a painted step to the deeper ground, held | none | No |
| (route) | /corrections | `grow`: the spine draws downward | `flow+in` once on the rows | Spine yes |

**Checks against Step 2 of the skill:**

- Device families used: kinetic, pointer, colour sweep, count, reveal, grow,
  seed assembly, flow+in, type, ground. **Ten, against a floor of four.**
- No family is an act's reason twice in a row. Reading the reason column down:
  kinetic, colour sweep, seed assembly, flow+in, type, ground.
- `scrub` acts: **zero**, and that is a constraint rather than a choice. No
  footage, no asset generation.
- No two adjacent acts carry the same feeling.
- The peak act has the most scroll room on the page by a visible margin, and the
  authored silence sits directly in front of it.
- Total page length: measured, not asserted. See the final report.
- The 6-to-7-acts-at-13.6-to-13.8vh fingerprint band is not hit: this page is
  six acts at roughly 6 viewport-heights, because it is an editorial page and the
  skill says a short grammar should stay short rather than padding to a quota.

---

## The deviation from the skill, recorded

**`engine/scrollcraft.js` is not adopted.** It is 1,283 lines of vanilla DOM
that would duplicate anime.js v4, already installed here, and would fight React
19's hydration on a Next.js static export. The skill's own rule is that bespoke
behaviour lives in the page rather than in the engine, so this build applies the
skill's principles (the variety law, the feeling curve, the peak, the taste
floor, the device families) on the existing anime.js stack. Every `data-sc-*`
attribute in the references is therefore a pattern to re-implement, not an
attribute to write.
