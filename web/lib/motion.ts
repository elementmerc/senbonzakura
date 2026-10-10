// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

import { animate, createTimeline, onScroll, stagger, utils } from 'animejs';

/**
 * WHY THIS FILE HOLDS TEN DEVICES AND NOT ONE.
 *
 * It used to hold one: fade in, rise fourteen pixels, ease out. That is the right fade, and it was
 * applied to all five sections, so the page read as static however correct each individual
 * animation was. Five sections that behave identically are one section shown five times, and the
 * eye works that out in about two seconds.
 *
 * So the devices below are deliberately different KINDS of movement rather than one movement at
 * different sizes. The rule the page is built to: at least four distinct families, and no family
 * is the reason an act exists twice in a row. Making the existing fade bigger or slower would have
 * been the obvious fix and the wrong one, because restraint is the signal; what was missing was
 * variety.
 *
 * Which device carries which section is written down, once, in
 * `scrollcraft/builds/senbon-dev/BRIEF.md`. Read that before moving one.
 *
 * THE ONE SPLIT THAT RUNS THROUGH THE WHOLE FILE. A figure re-derives itself every time you look
 * at it, so figures and graphics are tied to scroll and run backwards when the reader goes back up.
 * Prose does not: text that un-reveals when somebody scrolls up to re-read a sentence is a defect
 * dressed as an effect. So `revealOnce` fires once and never hides again, and everything that
 * draws a number is scrubbed. Before this split every paragraph on the page measured opacity 0
 * again on the way back to the top.
 */

/**
 * WHY THIS FILE WAS REWRITTEN, because the first version animated nothing at all.
 *
 * It used `opacity: [0, 1]`, which is the anime.js v3 idiom. In v4 a bare value is the
 * DESTINATION, not the journey, and the start value goes in a tween parameter object:
 * `opacity: { from: 0 }`. The array form is what almost every example on the web uses and it is
 * not honoured here, so every animation resolved to "move to where you already are" and did
 * nothing. The `./results/` panel was the visible proof: its rows carried an inline `opacity: 0`
 * waiting for an animation that could never arrive, so the panel rendered empty.
 *
 * The easings were wrong in the same way. `ease: 'outExpo'` is a v3 name; v4 takes `'out(3)'` and
 * friends, and an unrecognised easing fails SILENTLY as linear rather than erroring.
 *
 * THE RULE THAT PREVENTS THE WHOLE CLASS: never skip an animation, give it `duration: 0`. An
 * element parked at its start value by an animation that never ran is invisible content, which is
 * the single most common way animation breaks a page, and it is what happened here.
 */

/**
 * THE SCROLL RANGE, and the reason it is short.
 *
 * `sync` maps an animation's whole timeline onto the scroll distance between `enter` and `leave`,
 * so those two thresholds are not a trigger, they are the DURATION. The first version used the
 * defaults, which run from the element appearing at the bottom of the viewport to the element
 * disappearing off the top: the full journey. An animation stretched over that is at about 10%
 * progress when the element is comfortably readable, so a reader sees a half-arrived page and
 * concludes nothing is animating. That is exactly what happened, and the refusal grid was the
 * proof: four hundred squares still at their start opacity while sitting in the middle of the
 * screen.
 *
 * So the range ends when the element's top reaches the middle of the viewport. The movement
 * finishes as the element settles into reading position and reverses on the way back up, which
 * is the behaviour on animejs.com.
 *
 * The grammar is '<container edge> <target edge>', container FIRST, which is the opposite way
 * round from how it reads in English. Checked against the installed bundle rather than assumed:
 * `top`/`start` resolve to 0, `bottom`/`end` to 100%, `center` to 50%.
 */
const ENTER = 'end-=40 start';
const LEAVE = 'center start';

/**
 * A stagger step that keeps the WHOLE wave inside `spread`, however many elements there are.
 *
 * A fixed per-element delay is a trap at scale. `stagger(60)` across the refusal grid's 175 lit
 * cells is a ten-second wave, and under `sync` that ten seconds is the entire scroll range, so
 * the last cells arrive only once the grid is leaving the screen. The reader never sees the
 * finished figure. Bounding the total means a list of six and a grid of four hundred both take
 * about the same time, which is also what makes them feel like one page rather than two.
 */
function wave(count: number, { spread = 520, max = 70 } = {}): number {
  if (count < 2) return 0;
  return Math.min(max, spread / (count - 1));
}

/**
 * WHETHER THE SYSTEM'S REDUCED-MOTION SETTING IS HONOURED. Operator decision, 2026-10-10: it is
 * not, and this constant is the whole of that decision so it is one line to revert and a reviewer
 * cannot mistake it for an oversight.
 *
 * The usual argument for honouring it is strong and is written down here rather than deleted,
 * because it is the thing being traded away. The setting exists for readers who get motion
 * sickness or vestibular symptoms from movement they did not ask for, and a looping animation is
 * the worst case because it cannot be waited out. We now run everything regardless.
 *
 * What makes that defensible here rather than merely convenient: nothing on this page is a scroll
 * hijack, every scroll-linked animation is driven by the reader's own scrolling and stops the
 * moment they stop, and the two looping pieces (the mark's petals, the typing prompt) are small,
 * slow and off to one side rather than behind the text. What it costs is real all the same, and
 * the honest statement is that this page prioritises the demonstration over that setting.
 *
 * Flipping this back to true restores the previous behaviour everywhere at once.
 */
const RESPECT_SYSTEM_MOTION_SETTING = false;

/**
 * Whether an element is still below the line where its animation begins.
 *
 * THIS IS THE FIX FOR A WHOLE CLASS OF DEFECT, found by measuring the built page rather than by
 * reading it, and it is worth writing down because every scroll device here was quietly carrying
 * it.
 *
 * An animation that has not yet entered its scroll range has not been seeked, so the element sits
 * at whatever the markup and the stylesheet give it, which on this page is the FINISHED state. The
 * reader therefore meets a figure at its final value, and the moment they cross the trigger it
 * snaps back to the start and animates forward. Measured at 1440x900: the four figures in the gap
 * section read 43.8, 2.72, 21.8 and 67.8 at the top of the page and jumped to 11.1, 0.69, 5.5 and
 * 17.1 at scroll y=300. The drift bars stood at full height all the way down the page and dropped
 * to zero at y=1860 before growing back. Every individual frame of that looks correct, which is
 * why it survives being read and only shows up in a stepped scroll.
 *
 * It bites unevenly, which is what makes it confusing rather than obvious. A plain `animate()` over
 * elements applies its `from` values when it is created, so those devices were already right. A
 * TIMELINE does not apply the from-values of a child whose delay has not arrived, and an animation
 * whose target is a plain object has no element style to apply at all, so the counters and the
 * assembly were both wrong.
 *
 * So: anything scroll-linked primes its own start state, and only when the element is below the
 * line where its range begins. The condition matters as much as the priming. Priming
 * unconditionally would park any element the reader has already scrolled PAST at the start value
 * for good, because the observer never seeks an animation outside its range, and that is how a
 * reader who reloads halfway down a page ends up with a blank figure.
 */
function belowStart(el: Element, inset = 0): boolean {
  if (typeof window === 'undefined') return false;
  return el.getBoundingClientRect().top > window.innerHeight - inset;
}

/** Whether this reader has asked their system for less movement, and we are listening. */
export function prefersReducedMotion(): boolean {
  if (!RESPECT_SYSTEM_MOTION_SETTING) return false;
  return (
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );
}

/**
 * How long something should take, given the reader's preference.
 *
 * Zero rather than skipped, always. A reduced-motion reader gets the finished page instantly;
 * they do not get a page stuck at the start of an animation nobody ran.
 */
function ms(normal: number): number {
  return prefersReducedMotion() ? 0 : normal;
}

/**
 * THE PROSE DEVICE. Reveal elements as they arrive, once, and never hide them again.
 *
 * `flow` plus `in`, which is the plainest thing in the kit and the right thing for most of a page.
 * Opacity from zero, a rise of fourteen pixels, 620ms on an ease-out, children staggered so the
 * eye is led down the stack in reading order. Larger, slower or bouncier is not more premium; the
 * restraint is the signal.
 *
 * WHY AN INTERSECTION OBSERVER RATHER THAN A SCROLL-SYNCED ANIMATION, which is what this used to
 * be. Two reasons, and the second one is the one that was costing the page.
 *
 * One: a scroll-synced reveal runs backwards. That is correct for a chart and wrong for a
 * sentence, because somebody scrolling up to re-read a line should find the line there.
 *
 * Two, and this is the measured defect: one `animate()` call over eight elements gets ONE scroll
 * observer, and that observer watches the FIRST element. So a section taller than the viewport
 * animated its last paragraphs while they were still a screen below the fold, and by the time the
 * reader reached them they had already finished. The movement was real and nobody could see it,
 * which is a precise description of how this page came to feel static. One observer per element
 * fixes it, and an IntersectionObserver is the cheap way to have one each.
 *
 * The stagger is computed per BATCH rather than per element: whatever crosses the line in the same
 * callback is treated as one cluster and offset in document order, so a heading and its paragraph
 * arriving together still read as a wave, and a lone element a screen later does not inherit a
 * delay it has no reason to carry.
 *
 * The bottom margin pulls the trigger line 12% up from the fold, so the movement starts when the
 * reader is looking at the element rather than when its first pixel clears the bottom edge.
 */
export function revealOnce(selector: string, { rise = 14, step = 70 } = {}): void {
  const targets = utils.$(selector) as HTMLElement[];
  if (targets.length === 0) return;

  const still = prefersReducedMotion();
  const order = new Map(targets.map((el, i) => [el, i]));

  const play = (el: HTMLElement, delay: number) => {
    animate(el, {
      // EXPLICIT DESTINATIONS, because the start state is primed below rather than left to the
      // engine. A bare `{ from: 0 }` reads the element's CURRENT value as the destination, and a
      // primed element's current value is zero, so the pair would animate from nothing to nothing.
      opacity: { from: 0, to: 1 },
      // The rise is the part that travels, so it is the part the preference removes. The fade
      // stays, because a fade moves nothing across the screen and is not what the setting is for.
      ...(still ? {} : { translateY: { from: rise, to: 0 } }),
      duration: ms(620),
      delay: still ? 0 : delay,
      ease: 'out(3)',
      // The inline styles come back off, so nothing on the page is left pinned by a finished
      // animation and a later stylesheet change still reaches the element.
      onComplete: () => {
        el.style.opacity = '';
        el.style.transform = '';
      },
    });
  };

  if (typeof IntersectionObserver === 'undefined') {
    // No observer means no trigger, so everything plays at once rather than staying hidden. An
    // element parked at its start value by an animation that never ran is invisible content.
    targets.forEach((el) => play(el, 0));
    return;
  }

  // The trigger line sits 12% up from the fold, so the movement starts when the reader is looking
  // at the element rather than when its first pixel clears the bottom edge. Anything still below
  // that line is hidden now, so it is never seen at full strength and then snapped back to zero.
  // Measured before this: a paragraph was readable for about a hundred pixels of scroll and then
  // flickered out and faded in again.
  const inset = typeof window === 'undefined' ? 0 : window.innerHeight * 0.12;
  targets.forEach((el) => {
    if (!still && belowStart(el, inset)) el.style.opacity = '0';
  });

  const observer = new IntersectionObserver(
    (entries) => {
      const arrived = entries
        .filter((e) => e.isIntersecting)
        .map((e) => e.target as HTMLElement)
        .sort((a, b) => (order.get(a) ?? 0) - (order.get(b) ?? 0));

      arrived.forEach((el, i) => {
        observer.unobserve(el);
        play(el, i * step);
      });
    },
    { rootMargin: '0px 0px -12% 0px', threshold: 0 },
  );

  targets.forEach((el) => observer.observe(el));
}

/**
 * THE GRID DEVICE. A sweep of colour across a field of cells, tied to scroll, so it colours up on
 * the way in and colours back down on the way out.
 *
 * This is NOT the prose fade with a different selector, and the difference is in the markup rather
 * than here: the refusal grid draws all four hundred cells grey and lays the hundred and
 * seventy-five pink ones on top, so a pink cell fading in reads as that square TURNING pink rather
 * than as a square appearing out of nothing. One animated property, opacity, which the compositor
 * handles without touching layout, and both endpoints stay theme tokens so the sweep is correct in
 * either theme and survives the toggle.
 *
 * `target` is passed explicitly and is the GRID, not a cell. With one animation over a hundred and
 * seventy-five elements there is one observer, and left to itself it watches the first element: the
 * whole grid would then be driven by the position of its top-left square. Naming the container
 * makes the sweep belong to the figure.
 */
export function colourSweep(
  container: HTMLElement,
  selector: string,
  { spread = 700 } = {},
): void {
  const cells = Array.from(container.querySelectorAll<HTMLElement>(selector));
  if (cells.length === 0) return;

  const still = prefersReducedMotion();

  animate(cells, {
    opacity: { from: 0 },
    duration: ms(520),
    delay: still ? 0 : stagger(wave(cells.length, { spread, max: 40 })),
    ease: 'out(2)',
    autoplay: onScroll({
      target: container,
      enter: ENTER,
      leave: LEAVE,
      sync: still ? false : 2,
      repeat: true,
    }),
  });
}

/**
 * Count a number up to its value, tied to scroll so it also counts back down.
 *
 * The plain object is animated rather than the element, so the formatting stays ours and the
 * precision survives: 43.8 is a different claim from 44, and rounding a cited figure inside an
 * animation would misquote it.
 *
 * The element's text already holds the final value in the markup, so a reader with no JavaScript,
 * or one whose scroll trigger never fires, sees the correct number rather than a zero.
 */
export function countUp(
  el: Element,
  to: number,
  { decimals = 0, duration = 1100 } = {},
): void {
  const state = { value: 0 };
  // PRIMED, because this animation's target is a plain object and so there is no element style for
  // the engine to apply a start value to. Without this the reader meets the final figure, and it
  // jumps back to zero the moment they cross the trigger. Only when the figure is still below the
  // line where its range begins: priming one the reader has already scrolled past would leave it
  // reading zero for good, because the observer never seeks an animation outside its range.
  if (belowStart(el, 40)) el.textContent = (0).toFixed(decimals);
  animate(state, {
    value: to,
    duration: ms(duration),
    ease: 'out(3)',
    onUpdate: () => {
      el.textContent = state.value.toFixed(decimals);
    },
    onComplete: () => {
      el.textContent = to.toFixed(decimals);
    },
    autoplay: onScroll({
      // `target` IS NOT OPTIONAL HERE, and leaving it out is why the numbers never counted.
      //
      // A ScrollObserver with no target asks the animation it is linked to for a DOM element to
      // watch. This animation's target is a plain object, deliberately, so the formatting stays
      // ours and 43.8 does not get rounded to 44 inside a tween. There is no element to find, so
      // the library falls back to `document.body`, whose box is the whole page: the counter was
      // being driven by total page scroll rather than by arriving on screen, so it inched up over
      // seven thousand pixels and was never seen to move.
      target: el as HTMLElement,
      enter: ENTER,
      leave: LEAVE,
      sync: prefersReducedMotion() ? false : 3,
      repeat: true,
      // RESET EXPLICITLY WHEN THE RANGE IS LEFT GOING BACKWARDS. Measured: scrolling down and
      // back up left every counter showing its final figure. An element animation does not have
      // this problem, because the observer seeks the animation to zero and the browser repaints
      // the style; here the number only exists because onUpdate wrote it, and once the range is
      // behind you there are no more updates to write it back. So the boundary says so directly.
      onLeaveBackward: () => {
        el.textContent = (0).toFixed(decimals);
      },
    }),
  });
}

/**
 * Grow a set of elements from nothing, tied to scroll. Used by the bars and the timeline spine.
 *
 * `transformOrigin` is the caller's job, in CSS, because a bar grows from its base and a spine
 * grows from its top and this function cannot know which.
 *
 * THE RANGE IS AN ARGUMENT BECAUSE THE DEFAULT ONE IS WRONG FOR A TALL ELEMENT. The page's usual
 * range runs from the element's top nearing the fold to its top reaching the middle of the screen,
 * which is about half a viewport whatever the element is. That is right for a bar in a chart and
 * useless for the corrections spine, which is two and a half thousand pixels tall: measured, it
 * finished drawing by scroll y=265 of 1755 and then stood still for the rest of the page, so the
 * one graphic on that page whose motion carries its meaning, a sequence going downward, had
 * stopped moving before the reader read the first entry.
 */
export function growIn(
  targets: Element[] | string,
  {
    axis = 'scaleY',
    duration = 900,
    step = 110,
    enter = ENTER,
    leave = LEAVE,
  }: {
    axis?: 'scaleX' | 'scaleY';
    duration?: number;
    step?: number;
    enter?: string;
    leave?: string;
  } = {},
): void {
  const list = typeof targets === 'string' ? utils.$(targets) : targets;
  if (list.length === 0) return;
  const still = prefersReducedMotion();

  animate(list, {
    [axis]: { from: 0 },
    duration: ms(duration),
    delay: still ? 0 : stagger(Math.min(step, wave(list.length, { spread: 700, max: step }))),
    ease: 'out(3)',
    autoplay: onScroll({
      enter,
      leave,
      sync: still ? false : 2,
      repeat: true,
    }),
  });
}

/**
 * The two floating petals drift, independently and slowly.
 *
 * THE ONE ANIMATION THAT STILL STOPS ENTIRELY under reduced motion, and the only one that loops.
 * A loop is the clearest vestibular trigger there is: it never ends, so a reader cannot wait it
 * out. Everything else on this page either fades, which travels nowhere, or is tied to a scroll
 * the reader is driving themselves.
 *
 * Returns a stop function; a loop left running after its component is gone keeps the compositor
 * awake, which is somebody's battery.
 */
export function driftPetals(root: Element): () => void {
  if (prefersReducedMotion()) return () => {};

  const petals = Array.from(root.querySelectorAll<SVGElement>('[data-petal]'));
  if (petals.length === 0) return () => {};

  const running = petals.map((petal, i) => {
    const sign = petal.dataset.petal?.startsWith('-') ? -1 : 1;
    return animate(petal, {
      // Translation only, inside the parent that carries the petal's rotation, so the petal
      // drifts out along its own axis and back. A rotation here would need a transform origin to
      // be right and would be a rotation of part of the mark, which the brand rules are about.
      translateY: { from: 0, to: -3.4 },
      translateX: { from: 0, to: sign * 1.2 },
      duration: 3600 + i * 950,
      delay: i * 850,
      ease: 'inOut(2)',
      loop: true,
      alternate: true,
    });
  });

  return () => running.forEach((a) => a.pause());
}

/**
 * Type a whole block out, line after line, hold it, wipe it, and start again.
 *
 * NOT ONE LINE. The first version typed a single prompt and left the rest of the panel sitting
 * there as static text, which is not what a terminal looks like when something is running: the
 * command goes in, then the output arrives underneath it, line by line. A caret blinking above
 * five lines that were always there is a blinking caret, not a terminal.
 *
 * Each line keeps its own element, so the filenames stay in the accent colour and the notes stay
 * grey. Typing one flat string into a `<pre>` would be simpler and would throw the colour away,
 * which is most of what makes the panel readable.
 *
 * The real text stays in the markup and this clears it on mount. A reader whose JavaScript never
 * arrives sees the finished listing; only a reader who is definitely getting the animation ever
 * sees it empty.
 *
 * Returns a stop function. A loop still running after its component has gone keeps the
 * compositor awake, which is somebody's battery.
 */
export function typeTranscript(
  lines: Element[],
  { perChar = 26, lineGap = 90, hold = 2200, wipe = 420 } = {},
): () => void {
  // Paired up front, so indexing never has to be proved safe twice: a line and its text travel
  // together and the tween closes over the pair rather than over an index into two arrays.
  const rows = lines.map((el) => ({ el, text: el.textContent ?? '' }));
  if (rows.length === 0) return () => {};

  const paint = (row: { el: Element; text: string }, n: number) => {
    row.el.textContent = row.text.slice(0, Math.max(0, Math.round(n)));
  };
  rows.forEach((row) => paint(row, 0));

  const tl = createTimeline({ loop: true });
  rows.forEach((row, i) => {
    const state = { n: 0 };
    tl.add(
      state,
      {
        n: { from: 0, to: row.text.length },
        duration: Math.max(140, row.text.length * perChar),
        ease: 'linear',
        onUpdate: () => paint(row, state.n),
      },
      i === 0 ? undefined : `+=${lineGap}`,
    );
  });

  // Hold the finished block, then clear every line at once rather than un-typing each one.
  // Backspacing five lines takes as long as typing them and the reader has already read it; a
  // terminal being cleared is one action, not five.
  const done = { n: 1 };
  tl.add(done, { n: 0, duration: hold, onComplete: () => rows.forEach((row) => paint(row, 0)) })
    .add(done, { n: 1, duration: wipe });

  return () => tl.pause();
}

/**
 * THE HERO DEVICE. Type that assembles, one line at a time, out from behind a mask.
 *
 * `kinetic`, and lines rather than characters. Splitting display type per character turns reading
 * into waiting, and the one place it is ever right is a page whose whole subject is the
 * typography. Lines are almost always the answer.
 *
 * THE LINES ARE IN THE MARKUP, not measured here, and that is deliberate rather than lazy. The
 * usual implementation wraps every word in a span, reads the real line boxes, groups by
 * `offsetTop` and re-runs after `document.fonts.ready`. On a headline that already carries a hard
 * break and a coloured second half, that rebuild would have to reconstruct the colour, and a
 * measurement that runs before the webfont lands groups the words wrongly. This headline has
 * exactly two lines by authorial decision, so the two masks are written out and there is nothing
 * to measure, nothing to race, and no font-loading order to get right.
 *
 * It plays once, on mount, with no scroll trigger. A hero cue that ramps up from nothing as the
 * reader scrolls means the landing view, the one screen every visitor sees, has no headline on it.
 */
export function riseLines(selector: string, { step = 90 } = {}): void {
  const lines = utils.$(selector) as HTMLElement[];
  if (lines.length === 0) return;

  const still = prefersReducedMotion();

  if (still) {
    animate(lines, { opacity: { from: 0, to: 1 }, duration: 0 });
    return;
  }

  // THE TRAVEL IS MEASURED IN PIXELS, ONE LINE HEIGHT, AND NOT WRITTEN AS A PERCENTAGE.
  //
  // `translateY: { from: '115%' }` is the obvious way to say "start one line below your own mask",
  // and it did not do that. Measured in Chromium: the line boxes are 88px tall, and the computed
  // transform at the start of the animation was about 866px, so the unit was resolving against
  // something else entirely. The mask is 100px tall, so the line was invisible for the first nine
  // tenths of its journey and then whipped into place in the last eighty milliseconds. It reads as
  // a late pop rather than as a rise, and it looks completely correct in a screenshot.
  //
  // So the distance is read off the element. One line height is the right distance by definition:
  // any less and the line is already peeking out of the mask before it moves, any more and the
  // reader waits for it.
  lines.forEach((line, i) => {
    const travel = line.offsetHeight || 80;
    animate(line, {
      translateY: { from: travel, to: 0 },
      duration: ms(780),
      delay: i * step,
      ease: 'out(3)',
    });
  });
}

/**
 * Publish the pointer's position on an element as `--mx` and `--my`, in percent.
 *
 * The page's one piece of embodiment: a wash of the accent, very faint, that follows the reader
 * across the hero. It exists so the first screen responds to somebody being there rather than
 * playing a film at them, and it is the reason the hero can afford to hold still otherwise.
 *
 * Interpolated toward the pointer rather than tracking it. Direct tracking carries no momentum and
 * reads as artificial, which is the opposite of the point.
 *
 * Gated to a real pointer. A touch screen fires a single synthetic hover on tap, so an ungated
 * version lights up once in the wrong place and then never moves again.
 *
 * Returns a stop function. A rAF loop left running after its component has gone keeps the
 * compositor awake, which is somebody's battery.
 */
export function pointerWash(el: HTMLElement): () => void {
  if (typeof window === 'undefined') return () => {};
  if (!window.matchMedia('(hover: hover) and (pointer: fine)').matches) return () => {};
  if (prefersReducedMotion()) return () => {};

  let targetX = 50;
  let targetY = 40;
  let x = 50;
  let y = 40;
  let frame = 0;
  let moved = false;

  const onMove = (event: PointerEvent) => {
    const box = el.getBoundingClientRect();
    if (box.width === 0 || box.height === 0) return;
    targetX = ((event.clientX - box.left) / box.width) * 100;
    targetY = ((event.clientY - box.top) / box.height) * 100;
    moved = true;
  };

  const tick = () => {
    x += (targetX - x) * 0.08;
    y += (targetY - y) * 0.08;
    el.style.setProperty('--mx', `${x.toFixed(2)}%`);
    el.style.setProperty('--my', `${y.toFixed(2)}%`);
    // Only lit once the reader has actually moved a pointer, so the wash is a response rather
    // than a decoration that was there all along.
    el.style.setProperty('--wash', moved ? '1' : '0');
    frame = requestAnimationFrame(tick);
  };

  window.addEventListener('pointermove', onMove, { passive: true });
  frame = requestAnimationFrame(tick);

  return () => {
    window.removeEventListener('pointermove', onMove);
    if (frame) cancelAnimationFrame(frame);
  };
}

/**
 * Drift an element toward the pointer when the pointer is near it.
 *
 * One element on the page gets this, the hero's pill, because it is the primary call to action. A
 * page of magnetic elements is a page you cannot click anything on.
 *
 * `magnet` and any scroll-driven transform cannot share an element: this writes `transform` every
 * frame in its own loop, so it silently wins and the other animation's work is discarded. The pill
 * therefore has no reveal of its own.
 *
 * Returns a stop function, for the reason given on `pointerWash`.
 */
export function magnet(el: HTMLElement, strength = 0.26, radius = 180): () => void {
  if (typeof window === 'undefined') return () => {};
  if (!window.matchMedia('(hover: hover) and (pointer: fine)').matches) return () => {};
  if (prefersReducedMotion()) return () => {};

  let targetX = 0;
  let targetY = 0;
  let x = 0;
  let y = 0;
  let frame = 0;

  const onMove = (event: PointerEvent) => {
    const box = el.getBoundingClientRect();
    const dx = event.clientX - (box.left + box.width / 2);
    const dy = event.clientY - (box.top + box.height / 2);
    const distance = Math.hypot(dx, dy);
    // Outside the radius it returns to rest rather than leaning toward a pointer on the far side
    // of the page, which is what makes it read as magnetism rather than as a wobble.
    const pull = distance > radius ? 0 : strength * (1 - distance / radius);
    targetX = dx * pull;
    targetY = dy * pull;
  };

  const tick = () => {
    x += (targetX - x) * 0.12;
    y += (targetY - y) * 0.12;
    el.style.transform = `translate(${x.toFixed(2)}px, ${y.toFixed(2)}px)`;
    frame = requestAnimationFrame(tick);
  };

  window.addEventListener('pointermove', onMove, { passive: true });
  frame = requestAnimationFrame(tick);

  return () => {
    window.removeEventListener('pointermove', onMove);
    if (frame) cancelAnimationFrame(frame);
    el.style.transform = '';
  };
}

/**
 * A wipe from one edge, once. `clip-path` eating across an element.
 *
 * A wipe reads as a change of state, which is what a result arriving is. A fade reads as something
 * loading. The table of our own numbers therefore wipes rather than fades, and it wipes ONCE and
 * stays, because a table is something somebody reads and reading must not re-hide.
 *
 * `clipPath` is written from `onUpdate` rather than handed to the engine as a property, because
 * interpolating between two `inset()` strings is not something to rely on, and a silently
 * unanimated property is the exact failure this file exists to document. A plain object tween with
 * a string written per frame cannot fail that way.
 *
 * `clip-path` is relative to the border box, not to the ink, so this belongs on a wrapper or on an
 * element with room around its type. On a heading set with a line height below one it would shear
 * the ascenders off.
 */
export function wipeInOnce(
  selector: string,
  { duration = 620, step = 110, edge = 'right' as 'right' | 'left' } = {},
): void {
  const targets = utils.$(selector) as HTMLElement[];
  if (targets.length === 0) return;

  const still = prefersReducedMotion();
  const order = new Map(targets.map((el, i) => [el, i]));

  const play = (el: HTMLElement, delay: number) => {
    if (still) {
      el.style.clipPath = '';
      return;
    }
    const state = { w: 100 };
    const paint = () => {
      const inset = edge === 'right' ? `inset(0 ${state.w}% 0 0)` : `inset(0 0 0 ${state.w}%)`;
      el.style.clipPath = inset;
    };
    paint();
    animate(state, {
      w: 0,
      duration,
      delay,
      ease: 'out(3)',
      onUpdate: paint,
      // The clip is removed rather than left at zero. A stale `inset(0 0 0 0)` still establishes a
      // clipping box, which quietly cuts off anything that later overflows the element, and a
      // focus ring on a link inside the row is exactly such a thing.
      onComplete: () => {
        el.style.clipPath = '';
      },
    });
  };

  if (typeof IntersectionObserver === 'undefined') {
    targets.forEach((el) => play(el, 0));
    return;
  }

  // Clipped now rather than on arrival, for the same reason the reveal is faded now: a row that is
  // readable for a tenth of a viewport and then wipes in from nothing reads as the page correcting
  // itself.
  const inset = typeof window === 'undefined' ? 0 : window.innerHeight * 0.1;
  targets.forEach((el) => {
    if (!still && belowStart(el, inset)) {
      el.style.clipPath = edge === 'right' ? 'inset(0 100% 0 0)' : 'inset(0 0 0 100%)';
    }
  });

  const observer = new IntersectionObserver(
    (entries) => {
      const arrived = entries
        .filter((e) => e.isIntersecting)
        .map((e) => e.target as HTMLElement)
        .sort((a, b) => (order.get(a) ?? 0) - (order.get(b) ?? 0));
      arrived.forEach((el, i) => {
        observer.unobserve(el);
        play(el, i * step);
      });
    },
    { rootMargin: '0px 0px -10% 0px', threshold: 0 },
  );

  targets.forEach((el) => observer.observe(el));
}

/**
 * THE SIGNATURE MOVE. The chart assembles itself out of the individual runs it is the mean of.
 *
 * Five hairline ticks land at their real measured heights, in seed order, and only then does the
 * bar grow to the mean of them. Scroll back up and the ticks lift off and the bar retracts, which
 * is the whole claim of the page in one gesture: a measurement re-derives itself every time you
 * look at it.
 *
 * WHY THIS AND NOT A PRETTIER THING. Every height here is a real number, committed at
 * `evidence/k-sweep-2026-08-13/drift-per-seed.json` and already cited on the page. The two-direction
 * arm has one run at 0.1784 against a next-worst of 0.0828, and that single unlucky run is most of
 * why our own headline ratio is 1.9. A reader watching the bar assemble sees the seed spread
 * happen rather than being told about it in a column they will not read. The flattering version of
 * this chart is two bars that fade in, and we already had it.
 *
 * A timeline rather than two animations, because the order is the point: the evidence lands, then
 * the summary is drawn from it. Two independent animations would race.
 */
export function seedAssembly(container: HTMLElement): void {
  const bars = Array.from(container.querySelectorAll<HTMLElement>('[data-bar]'));
  if (bars.length === 0) return;

  // Each bar is paired with the ticks it is the mean of, by the column they share, so the sequence
  // can run arm by arm: one arm's five runs land, then that arm's bar is drawn out of them, then
  // the next arm. Collecting all ten ticks in one list and staggering them would put both arms'
  // evidence on screen before either summary, which is a slideshow rather than an argument.
  const arms = bars
    .map((bar) => ({
      bar,
      ticks: Array.from(bar.parentElement?.querySelectorAll<HTMLElement>('[data-seed]') ?? []),
    }))
    .filter((arm) => arm.ticks.length > 0);
  if (arms.length === 0) return;

  const still = prefersReducedMotion();

  // PRIMED, AND THIS IS THE ONE THAT WAS WORST. A timeline does not apply the start value of a
  // child whose position has not arrived, so on a first read the chart stood fully assembled, bars
  // at full height and every tick lit, all the way down the page, and then dropped to nothing and
  // rebuilt itself once the reader crossed the trigger. Measured at 1440x900: bars at scaleY 1
  // from load to scroll y=1800, then 0.266 and 0 at y=1860.
  //
  // The resting opacity is read off the element rather than written here, because it lives in the
  // stylesheet beside the colour and duplicating the number in two files is how the two drift
  // apart. With the start state primed, every destination below has to be stated outright: a bare
  // `from` would read the primed zero as the destination and animate nothing to nothing.
  const rest = new Map<HTMLElement, number>();
  arms.forEach((arm) => {
    arm.ticks.forEach((tick) => rest.set(tick, parseFloat(getComputedStyle(tick).opacity) || 1));
  });
  if (!still) {
    arms.forEach((arm) => {
      arm.ticks.forEach((tick) => {
        tick.style.opacity = '0';
        tick.style.transform = 'scaleX(0.18)';
      });
      arm.bar.style.transform = 'scaleY(0)';
    });
  }

  const timeline = createTimeline({
    autoplay: onScroll({
      target: container,
      enter: ENTER,
      // A longer range than the page's default, because this is the peak and an assembly the
      // reader cannot follow is a flourish rather than an argument. It finishes when the chart's
      // top is a third of the way up the viewport rather than halfway.
      leave: 'center-=18% start',
      sync: still ? false : 2,
      repeat: true,
    }),
  });

  // ABSOLUTE POSITIONS IN MILLISECONDS, which is a number and needs no position-string grammar to
  // be right. The relative forms are terser and one wrong token in them fails by putting an
  // animation somewhere plausible rather than by erroring, which on a scroll-synced timeline is
  // invisible in every screenshot.
  const TICK = 110;
  const ARM = 780;
  arms.forEach((arm, a) => {
    const base = a * ARM;
    arm.ticks.forEach((tick, i) => {
      timeline.add(
        tick,
        {
          opacity: { from: 0, to: rest.get(tick) ?? 1 },
          ...(still ? {} : { scaleX: { from: 0.18, to: 1 } }),
          duration: ms(340),
          ease: 'out(3)',
        },
        base + i * TICK,
      );
    });
    timeline.add(
      arm.bar,
      {
        // Height, never opacity, when the motion is on: the bar is translucent at rest so its own
        // seed ticks stay visible through it, and tweening opacity toward that resting value would
        // flash the bar brighter on the way. Under the preference it fades in instead, because
        // there is no growth to carry it.
        ...(still
          ? { opacity: { from: 0, to: 0.88 } }
          : { scaleY: { from: 0, to: 1 } }),
        duration: ms(700),
        ease: 'out(3)',
      },
      base + arm.ticks.length * TICK + 40,
    );
  });
}

/**
 * The marker on a drawn interval, walking to where the measurement landed.
 *
 * The 1.9 figure is printed with its 95% interval beside it, because the docs page records that
 * giving it as a bare range once invited readers to mistake a sensitivity check for an interval.
 * This draws that interval rather than restating it: the band is 1.2 to 3.1 in real geometry, and
 * the marker walks in from the low end and comes to rest at 1.9. Scroll back up and it walks out.
 *
 * `transform` rather than `left`, which means the travel has to be in pixels, which means the
 * rail's width has to be known. It is measured once and re-measured on resize, because a value
 * cached at mount is wrong the moment somebody turns a phone sideways and every frame after.
 *
 * Returns a stop function, so the resize observer does not outlive the component.
 */
export function intervalMarker(
  marker: HTMLElement,
  rail: HTMLElement,
  { low, high, value }: { low: number; high: number; value: number },
): () => void {
  const span = high - low;
  if (span <= 0) return () => {};

  const fraction = Math.min(1, Math.max(0, (value - low) / span));
  let width = rail.clientWidth;
  const state = { at: 0 };

  const paint = () => {
    marker.style.transform = `translateX(${(state.at * width).toFixed(2)}px)`;
  };
  paint();

  animate(state, {
    at: fraction,
    duration: ms(900),
    ease: 'out(3)',
    onUpdate: paint,
    autoplay: onScroll({
      target: rail,
      enter: ENTER,
      leave: LEAVE,
      sync: prefersReducedMotion() ? false : 3,
      repeat: true,
      // Same boundary problem as the counters: this position only exists because `onUpdate` wrote
      // it, so once the range is behind you going backwards there are no more updates to write it
      // back to the start.
      onLeaveBackward: () => {
        state.at = 0;
        paint();
      },
    }),
  });

  if (typeof ResizeObserver === 'undefined') return () => {};
  const observer = new ResizeObserver(() => {
    width = rail.clientWidth;
    paint();
  });
  observer.observe(rail);
  return () => observer.disconnect();
}
