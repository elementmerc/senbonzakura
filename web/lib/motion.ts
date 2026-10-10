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

  if (prefersReducedMotion()) {
    utils.set(targets, { opacity: 1, translateY: 0 });
    return;
  }

  animate(targets, {
    opacity: [0, 1],
    translateY: [14, 0],
    duration: 760,
    delay: stagger(70, { start: delay }),
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

  if (prefersReducedMotion()) {
    show(to);
    return;
  }

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
