// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

import { animate, createTimeline, onScroll, stagger, utils } from 'animejs';

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

/** Whether this reader has asked their system for less movement. */
export function prefersReducedMotion(): boolean {
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
 * Reveal elements as they scroll in, and UNREVEAL them as they scroll back out.
 *
 * `sync` ties playback to scroll position rather than firing once, which is what makes the
 * movement reversible: scroll down and it plays, scroll up and it runs backwards. The reader can
 * still select the text and click the links while it does, because nothing here is a scroll
 * hijack; the page scrolls normally and the animation reads the position.
 *
 * `sync` takes a NUMBER as well as a boolean, and the number is the smoothing: a little lag so
 * the motion is not welded to the scrollbar, which is the difference between this feeling
 * designed and feeling like a slider. There is no separate smoothing PARAMETER, whatever the
 * ScrollObserver class exposes on itself; the accepted keys are in the installed type.
 */
export function reveal(selector: string, { rise = 18 } = {}): void {
  const targets = utils.$(selector);
  if (targets.length === 0) return;

  const still = prefersReducedMotion();

  animate(targets, {
    opacity: { from: 0 },
    // The rise is the part that travels, so it is the part the preference removes. The fade
    // stays, because a fade moves nothing across the screen and is not what the setting is for.
    ...(still ? {} : { translateY: { from: rise } }),
    duration: ms(620),
    delay: still ? 0 : stagger(wave(targets.length)),
    ease: 'out(3)',
    autoplay: onScroll({
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
 */
export function growIn(
  targets: Element[] | string,
  { axis = 'scaleY', duration = 900, step = 110 }: { axis?: 'scaleX' | 'scaleY'; duration?: number; step?: number } = {},
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
      enter: ENTER,
      leave: LEAVE,
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
 * Type a line out, hold it, wipe it, and move to the next. Loops.
 *
 * The one looping animation on the page besides the petals, and it is here because a panel
 * showing what a run leaves on disk should look like something that ran. A caret that never
 * moves is a screenshot.
 *
 * STOPS ENTIRELY under reduced motion rather than running at zero duration, which is the
 * opposite of the rule everywhere else in this file and is correct for exactly the same reason.
 * The rule exists so an element is never stranded at its start value; a loop has no final state
 * to be stranded short of, and a thing that retypes itself forever is the clearest vestibular
 * trigger there is because the reader cannot wait it out. So the element gets the longest phrase,
 * complete and still, and nothing moves.
 *
 * Returns a stop function. A loop still running after its component has gone keeps the
 * compositor awake, which is somebody's battery.
 */
export function typeLoop(
  el: Element,
  phrases: string[],
  { typeMs = 700, holdMs = 1400 } = {},
): () => void {
  if (phrases.length === 0) return () => {};
  if (prefersReducedMotion()) {
    el.textContent = phrases.reduce((a, b) => (b.length > a.length ? b : a));
    return () => {};
  }

  const tl = createTimeline({ loop: true });
  for (const phrase of phrases) {
    // One state object per phrase. Sharing one across the whole timeline makes each later tween
    // start from wherever the previous one left the value, which reads as a stutter.
    const state = { n: 0 };
    const paint = () => {
      el.textContent = phrase.slice(0, Math.round(state.n));
    };
    tl.add(state, { n: { from: 0, to: phrase.length }, duration: typeMs, ease: 'linear', onUpdate: paint })
      .add(state, { n: phrase.length, duration: holdMs, onUpdate: paint })
      .add(state, { n: { from: phrase.length, to: 0 }, duration: typeMs * 0.55, ease: 'linear', onUpdate: paint });
  }

  return () => tl.pause();
}
