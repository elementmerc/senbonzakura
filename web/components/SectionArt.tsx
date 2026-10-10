// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

'use client';

import { useEffect, useRef } from 'react';
import { growIn, reveal, typeLoop } from '@/lib/motion';

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

/** What a run writes out, as the five result files it leaves on disk. */
export function RunOutput() {
  const root = useRef<HTMLDivElement>(null);
  const FILES = [
    ['score.json', 'refusal rate, with an interval'],
    ['compass.json', 'does it separate harm from harmless'],
    ['coherence.json', 'what the edit cost in fluency'],
    ['capability.json', 'what it cost on tasks'],
    ['drift.json', 'how far the first token moved'],
  ];

  const prompt = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    // THE ROWS THIS PANEL RENDERED EMPTY. They carried an inline `opacity: 0` and waited for an
    // animation written in v3 syntax that could never run, so the panel showed its border and
    // its header and nothing else. The inline opacity is gone and anime sets the start value.
    reveal('[data-row]');

    // The panel showed what a run leaves behind and gave no sign that anything had run. A
    // command line that types itself, holds, wipes and types the next one is the smallest thing
    // that reads as a tool working rather than a screenshot of one. Every command here is real
    // and takes flags the CLI accepts; a decorative prompt that invents a command teaches the
    // reader something false.
    const el = prompt.current;
    if (!el) return;
    return typeLoop(el, [
      'senbonzakura measure ./model',
      'senbonzakura abliterate ./model --max-directions 2',
      'senbonzakura compare ./before ./after',
    ]);
  }, []);

  return (
    <div
      ref={root}
      aria-hidden="true"
      style={{
        width: '100%',
        border: '1px solid var(--rule)',
        borderRadius: '12px',
        overflow: 'hidden',
        background: 'var(--bg-raised)',
      }}
    >
      <div
        className="mono"
        style={{
          padding: '0.7rem 1.1rem',
          borderBottom: '1px solid var(--rule)',
          fontSize: '0.76rem',
          color: 'var(--ink-faint)',
          display: 'flex',
          gap: '0.5rem',
          whiteSpace: 'nowrap',
          overflow: 'hidden',
        }}
      >
        <span style={{ color: 'var(--measured)' }}>$</span>
        {/* The caret is a CSS pseudo-element on this span, so it sits flush against the last
            typed character without a space and without a node the typing has to work around. */}
        <span ref={prompt} className="caret" style={{ color: 'var(--ink-dim)' }} />
      </div>
      <div
        className="mono"
        style={{
          padding: '0.7rem 1.1rem',
          borderBottom: '1px solid var(--rule)',
          fontSize: '0.76rem',
          color: 'var(--ink-faint)',
        }}
      >
        ./results/
      </div>
      {FILES.map(([name, what]) => (
        <div
          key={name}
          data-row
          style={{
            display: 'flex',
            alignItems: 'baseline',
            gap: '0.9rem',
            padding: '0.72rem 1.1rem',
          }}
        >
          <span className="mono" style={{ fontSize: '0.82rem', color: 'var(--measured)' }}>
            {name}
          </span>
          <span style={{ fontSize: '0.78rem', color: 'var(--ink-faint)' }}>{what}</span>
        </div>
      ))}
    </div>
  );
}
