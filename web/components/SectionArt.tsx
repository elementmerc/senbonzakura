// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

'use client';

import { useEffect, useRef } from 'react';
import { growIn, reveal, typeTranscript } from '@/lib/motion';

/**
 * One piece of art per section, in the space the left-aligned prose leaves.
 *
 * EACH ONE DRAWS THAT SECTION'S OWN NUMBER and is not decoration standing in for an idea. The
 * grid really is 438 squares in 1000 because the measured refusal rate is 43.8%; the bars really
 * are in the ratio 0.0497 to 0.0932 because those are the two drift figures in the table beside
 * them. A chart that illustrates a number it does not come from is the thing this whole project
 * is against, and putting one on this page would be funny for the wrong reason.
 *
 * All of it is `aria-hidden`: every figure here is already stated in text, and a screen reader
 * announcing four hundred squares is worse than silence.
 */

/** 43.8% of system hardening requests refused, as 1,000 squares with 438 lit. */
export function RefusalGrid() {
  const root = useRef<HTMLDivElement>(null);
  const COLS = 25;
  const ROWS = 16;
  const TOTAL = COLS * ROWS; // 400 squares, so each is 0.25 of a percentage point.
  const LIT = Math.round(TOTAL * 0.438);

  useEffect(() => {
    // The lit cells fade in as the section arrives and fade back out on the way up, so the
    // figure reads as being counted rather than as a static picture.
    reveal('.cell-refused', { rise: 0 });
  }, []);

  return (
    <div ref={root} aria-hidden="true" style={{ width: '100%' }}>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${COLS}, 1fr)`,
          gap: '3px',
        }}
      >
        {Array.from({ length: TOTAL }, (_, i) => {
          const refused = i < LIT;
          // A class rather than an inline style on each of 400 nodes. The inline version put the
          // same five declarations into the HTML four hundred times and took the page from 48KB
          // to 104KB, on a page that makes a point of being small.
          return <span key={i} className={refused ? 'cell-refused' : 'cell-answered'} />;
        })}
      </div>
      <p
        className="mono"
        style={{
          marginTop: '1rem',
          fontSize: '0.74rem',
          color: 'var(--ink-faint)',
          letterSpacing: '0.04em',
        }}
      >
        400 squares. {LIT} are the refusals.
      </p>
    </div>
  );
}

/**
 * The two drift figures from the table, to scale.
 *
 * The bars are in the true ratio rather than a flattering one: 0.0932 against 0.0497 is 1.88, and
 * the taller bar is 1.88 times the shorter. Scaling the axis to make the gap look bigger is the
 * oldest chart lie there is.
 *
 * WHAT THE CHART WAS MISSING was not decoration, it was the caption a stranger needs to read it.
 * Two bars and four numbers, with the labels in the faintest ink on the page, left the reader to
 * work out what was being compared, which way was good, and which bar was which. So the chart now
 * says what it measures and which direction is better before the bars appear, the labels sit in
 * ordinary ink at a readable size, and the quieter bar is a visible grey rather than the hairline
 * rule colour, which at 15rem tall had almost disappeared.
 */
export function DriftBars() {
  const root = useRef<HTMLDivElement>(null);
  const BARS = [
    { label: 'One direction', value: 0.0497, spread: 0.0177 },
    { label: 'Two directions', value: 0.0932, spread: 0.0482 },
  ];
  const MAX = 0.14; // Headroom above the taller bar, so it does not touch the ceiling.

  useEffect(() => {
    const el = root.current;
    if (!el) return;
    growIn(Array.from(el.querySelectorAll('[data-bar]')), { axis: 'scaleY', duration: 950 });
    // The counters here are NOT registered by this component, deliberately. This chart renders
    // inside Measured's section, whose own effect already walks [data-count] across its whole
    // subtree, so registering them here put two competing animations on one node and the second
    // one read no data-decimals and formatted 0.0497 with zero decimal places. The figures
    // turned into "0". One owner per counter, and the owner is the section.
  }, []);

  return (
    <div ref={root} aria-hidden="true" style={{ width: '100%' }}>
      <p
        className="mono"
        style={{
          margin: '0 0 0.3rem',
          fontSize: '0.74rem',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          color: 'var(--ink-faint)',
        }}
      >
        Coherence drift
      </p>
      <p style={{ margin: '0 0 1.6rem', fontSize: '0.82rem', color: 'var(--ink-dim)' }}>
        How far the edit moved the model's ordinary writing. Lower is better.
      </p>
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-end',
          gap: '2.2rem',
          height: '15rem',
          paddingBottom: '0.6rem',
          borderBottom: '1px solid var(--rule)',
        }}
      >
        {BARS.map((b, i) => (
          <div key={b.label} style={{ flex: 1, textAlign: 'center' }}>
            <p
              className="mono"
              data-count={b.value}
              data-decimals="4"
              style={{ margin: '0 0 0.5rem', fontSize: '0.95rem', color: 'var(--ink)' }}
            >
              {b.value.toFixed(4)}
            </p>
            <div
              data-bar
              data-motion="travels"
              style={{
                height: `${(b.value / MAX) * 100}%`,
                minHeight: '4px',
                borderRadius: '5px 5px 0 0',
                transformOrigin: 'bottom',
                background: i
                  ? 'linear-gradient(to bottom, #F98DB0, #8B4791)'
                  : 'var(--ink-faint)',
              }}
            />
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: '2.2rem', marginTop: '0.7rem' }}>
        {BARS.map((b) => (
          <p
            key={b.label}
            style={{
              flex: 1,
              textAlign: 'center',
              margin: 0,
              fontSize: '0.92rem',
              color: 'var(--ink)',
              lineHeight: 1.6,
            }}
          >
            {b.label} removed
            <br />
            <span className="mono" style={{ fontSize: '0.76rem', color: 'var(--ink-faint)' }}>
              spread {b.spread.toFixed(4)} across 5 seeds
            </span>
          </p>
        ))}
      </div>
      <p
        style={{
          marginTop: '1.1rem',
          fontSize: '0.74rem',
          color: 'var(--ink-faint)',
          lineHeight: 1.6,
        }}
      >
        Drawn to scale. The taller bar is 1.88 times the shorter, which is what the figures say.
      </p>
    </div>
  );
}

/** What a run writes out, typed as a terminal transcript that loops. */
export function RunOutput() {
  const root = useRef<HTMLDivElement>(null);
  const FILES = [
    ['score.json', 'refusal rate, with an interval'],
    ['compass.json', 'does it separate harm from harmless'],
    ['coherence.json', 'what the edit cost in fluency'],
    ['capability.json', 'what it cost on tasks'],
    ['drift.json', 'how far the first token moved'],
  ];

  useEffect(() => {
    const el = root.current;
    if (!el) return;
    // Every line in reading order: the command, the directory it wrote, then one line per file.
    // The caret rides whichever line is being typed, which is the part that makes it read as a
    // terminal rather than as text fading in.
    const lines = Array.from(el.querySelectorAll<HTMLElement>('[data-line]'));
    return typeTranscript(lines);
  }, []);

  return (
    <div
      ref={root}
      aria-hidden="true"
      className="mono"
      style={{
        width: '100%',
        border: '1px solid var(--rule)',
        borderRadius: '12px',
        overflow: 'hidden',
        background: 'var(--bg-raised)',
        padding: '1.1rem 1.2rem 1.4rem',
        // A FIXED HEIGHT, because the panel empties itself every few seconds. Without one the
        // whole section would collapse and spring back on every loop, shoving the paragraph
        // beside it up and down the page, which is far worse than any animation is good.
        minHeight: '16.5rem',
        lineHeight: 1.9,
        fontSize: '0.82rem',
      }}
    >
      <div style={{ whiteSpace: 'pre', overflow: 'hidden' }}>
        <span style={{ color: 'var(--measured)' }}>$ </span>
        <span data-line style={{ color: 'var(--ink)' }}>
          senbonzakura measure ./model
        </span>
        <span className="caret" />
      </div>

      <div data-line style={{ color: 'var(--ink-faint)', whiteSpace: 'pre', overflow: 'hidden' }}>
        wrote ./results/
      </div>

      {FILES.map(([name, what]) => (
        <div key={name} style={{ whiteSpace: 'pre', overflow: 'hidden' }}>
          <span data-line style={{ color: 'var(--measured)' }}>{name}</span>
          <span data-line style={{ color: 'var(--ink-faint)' }}>{'   ' + what}</span>
        </div>
      ))}
    </div>
  );
}
