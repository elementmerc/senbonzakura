// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

import { animate, onScroll, stagger, utils } from 'animejs';

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
    delay: still ? 0 : stagger(60),
    ease: 'out(3)',
    autoplay: onScroll({
      enter: 'bottom-=80 top',
      leave: 'top bottom',
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
      enter: 'bottom-=60 top',
      leave: 'top bottom',
      sync: prefersReducedMotion() ? false : 3,
      repeat: true,
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
    delay: still ? 0 : stagger(step),
    ease: 'out(3)',
    autoplay: onScroll({
      enter: 'bottom-=60 top',
      leave: 'top bottom',
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
