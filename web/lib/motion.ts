// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

import { animate, stagger, utils } from 'animejs';

/**
 * Whether this reader has asked their system for less movement.
 *
 * Checked in JavaScript as well as CSS because an animation driven from JavaScript sets inline
 * styles, and an inline style wins against the stylesheet's reduced-motion rule. The CSS covers
 * what CSS drives; this covers the rest.
 *
 * WHAT THIS DOES AND DOES NOT TURN OFF, because the first version turned off everything and the
 * page then looked broken to anybody with the setting on, which is a lot of people who never
 * chose it deliberately.
 *
 * The setting exists for vestibular disorders, where MOVEMENT across the screen causes real
 * nausea and migraine. Translation, parallax, scaling, rotation and anything that loops are the
 * triggers. A fade between two opacities moves nothing, and a number counting up in place moves
 * nothing. So those keep running for everybody, and only the movement is dropped.
 *
 * The result is a page that is still visibly alive under the setting rather than one that looks
 * like the JavaScript failed, and nobody gets the motion that would hurt them.
 */
export function prefersReducedMotion(): boolean {
  return (
    typeof window !== 'undefined' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );
}

/**
 * Reveal elements, or just show them if the reader would rather nothing moved.
 *
 * The hidden-then-revealed pattern has one failure that matters: if the reveal never runs, the
 * content is invisible rather than merely unanimated. So the hidden state is applied by a rule
 * that depends on a class this module sets, and `reveal` sets opacity directly when motion is
 * declined. A reader with JavaScript off sees the page; a reader with reduced motion sees the
 * page; nobody sees nothing.
 */
export function reveal(selector: string, { delay = 0 } = {}): void {
  const targets = utils.$(selector);
  if (targets.length === 0) return;

  // The fade runs for everybody. Only the 14px rise is dropped, because that is the part that
  // moves and therefore the part the setting is about.
  const still = prefersReducedMotion();
  animate(targets, {
    opacity: [0, 1],
    ...(still ? {} : { translateY: [14, 0] }),
    duration: still ? 520 : 760,
    delay: stagger(still ? 50 : 70, { start: delay }),
    ease: 'outExpo',
  });
}

/**
 * Count a number up to its value, for a figure the page wants read rather than skimmed.
 *
 * `decimals` exists because these are measurements and a measurement's precision is part of it:
 * 43.8 is a different claim from 44, and rounding it in an animation would misquote a cited
 * figure. The element's text is set to the final value immediately when motion is declined, and
 * the markup carries the final value as its own content so the pre-animation state is correct
 * too.
 */
export function countUp(
  el: Element,
  to: number,
  { decimals = 0, duration = 1400, delay = 0 } = {},
): void {
  const show = (v: number) => {
    el.textContent = v.toFixed(decimals);
  };

  // A number counting up in place moves nothing across the screen, so it runs for everybody. It
  // is also the one animation on this page that carries meaning rather than polish: it makes a
  // reader look at the figure instead of skimming past it.
  const state = { value: 0 };
  animate(state, {
    value: to,
    duration,
    delay,
    ease: 'outExpo',
    onUpdate: () => show(state.value),
    onComplete: () => show(to),
  });
}

/**
 * Run `fn` the first time an element is on screen, then stop watching it.
 *
 * Entrance animations that fire on page load are finished before a reader scrolls to them, so a
 * block halfway down the page would animate to nobody. Where IntersectionObserver is missing, the
 * callback runs straight away rather than never, on the same fail-visible principle as above.
 */
export function onFirstView(
  el: Element,
  fn: () => void,
  { margin = '0px 0px -12% 0px' } = {},
): () => void {
  if (typeof IntersectionObserver === 'undefined') {
    fn();
    return () => {};
  }

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          fn();
          observer.disconnect();
        }
      }
    },
    { rootMargin: margin, threshold: 0.1 },
  );

  observer.observe(el);
  return () => observer.disconnect();
}

/**
 * The two floating petals drift, independently and slowly.
 *
 * The brand kit describes the mark as three large petals "with two independent smaller petals
 * floating in the gaps", so this animates what the mark already depicts rather than adding an
 * effect to it. The large petals and the mark's own silhouette, colour and proportions are left
 * alone, because the kit says not to rotate, stretch, recolour or add effects to the mark.
 *
 * Independently is the whole point: the two get different durations and a deliberate offset, so
 * they never beat together. Two petals moving in lockstep read as one mechanism, which is the
 * opposite of floating.
 *
 * Returns a stop function. A looping animation left running after its component is gone keeps the
 * compositor awake on a laptop, which is somebody's battery.
 */
export function driftPetals(root: Element): () => void {
  if (prefersReducedMotion()) return () => {};

  const petals = Array.from(root.querySelectorAll<SVGElement>('[data-petal]'));
  if (petals.length === 0) return () => {};

  const running = petals.map((petal, i) => {
    const sign = petal.dataset.petal?.startsWith('-') ? -1 : 1;
    return animate(petal, {
      // TRANSLATION ONLY, DELIBERATELY. These groups sit inside a parent that carries the
      // petal's `rotate(+/-25)` placement, so the movement happens in that rotated frame and
      // translateY drifts the petal along its own axis, out from the flower and back. A rotate
      // here would need a transform origin to be right and would be a rotation of part of the
      // mark, which the kit's rules are about. A drift out and back is the thing the mark
      // already depicts.
      translateY: [0, -3.4, 0],
      translateX: [0, sign * 1.2, 0],
      duration: 7200 + i * 1900,
      delay: i * 850,
      ease: 'inOutSine',
      loop: true,
    });
  });

  return () => running.forEach((a) => a.pause());
}
