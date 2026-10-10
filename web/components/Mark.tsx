// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

/**
 * Bloom, the project's mark, inlined so the two floating petals can be animated.
 *
 * The geometry, the gradients and their stops are copied verbatim from the brand kit's
 * `mark.svg`, which is also in `public/` for anything that wants the file. Inlining it is purely
 * so a script can reach the petal groups; nothing about the shape or the colour is changed, and
 * the gradient still runs pink to violet to navy top to bottom, which the kit says never to
 * recolour or flip.
 *
 * WHAT THE BRAND RULES ALLOW, read carefully rather than assumed. The kit says: do not rotate,
 * stretch, recolour or add effects to the mark. So the mark as a whole gets no rotation and no
 * scaling beyond its own size; its entrance is opacity only. The exception is the two SMALL
 * petals, which the kit itself describes as "two independent smaller petals floating in the
 * gaps". Drifting those a couple of degrees is the thing the mark is already depicting, and it
 * leaves the mark's silhouette, colour and proportions untouched. A reviewer who reads the rule
 * more strictly than this should say so; the drift lives in one place and comes out in one line.
 *
 * `aria-hidden` because the wordmark beside it already says "Senbonzakura", and a screen reader
 * announcing the name twice is worse than a decorative image going unannounced.
 */
export default function Mark({ size = 44 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 512 512"
      aria-hidden="true"
      focusable="false"
      data-mark
    >
      <defs>
        <linearGradient id="bloom-pn" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#F98DB0" />
          <stop offset="38%" stopColor="#E06A9C" />
          <stop offset="66%" stopColor="#8B4791" />
          <stop offset="100%" stopColor="#141A42" />
        </linearGradient>
        <linearGradient id="bloom-pns" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#F27FA6" />
          <stop offset="55%" stopColor="#7E4490" />
          <stop offset="100%" stopColor="#141A42" />
        </linearGradient>
        <linearGradient id="bloom-gloss" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#FFFFFF" stopOpacity=".6" />
          <stop offset="55%" stopColor="#FFFFFF" stopOpacity=".06" />
          <stop offset="100%" stopColor="#FFFFFF" stopOpacity="0" />
        </linearGradient>
      </defs>
      <g transform="translate(-30.034,63.595) scale(3.8138)">
        <g transform="translate(75,88)">
          {/* The three large petals, at the angles the kit sets. Untouched by any animation. */}
          {[-50, 50, 0].map((deg) => (
            <g key={deg} transform={`rotate(${deg})`}>
              <path
                d="M0,-8 C13,-28 13,-52 0,-70 C-13,-52 -13,-28 0,-8 Z"
                fill="url(#bloom-pn)"
              />
              <path
                d="M-3,-16 C3,-32 3,-52 -1,-62 C-7,-46 -8,-30 -3,-16 Z"
                fill="url(#bloom-gloss)"
              />
            </g>
          ))}
          {/* The two independent floating petals. These are what drifts.
             *
             * THREE NESTED GROUPS, AND THE NESTING IS LOAD-BEARING. A CSS transform on an SVG
             * element overrides that element's own `transform` ATTRIBUTE rather than composing
             * with it, so animating the group that carries `rotate(-25)` would snap the petal to
             * the centre of the flower the moment the animation started. The outer group holds
             * the placement the kit sets and is never animated, the middle group carries no
             * transform of its own and is what moves, and the inner group holds the radial
             * offset. */}
          {[-25, 25].map((deg) => (
            <g key={deg} transform={`rotate(${deg})`}>
              <g data-petal={deg}>
                <g transform="translate(0,-38)">
                  <path
                    d="M0,-3 C6,-15 6,-25 0,-30 C-6,-25 -6,-15 0,-3 Z"
                    fill="url(#bloom-pns)"
                  />
                </g>
              </g>
            </g>
          ))}
        </g>
      </g>
    </svg>
  );
}
