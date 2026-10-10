// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

'use client';

import { useEffect, useRef } from 'react';
import { colourSweep, seedAssembly, typeTranscript } from '@/lib/motion';

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
    const el = root.current;
    if (!el) return;
    // THE SQUARES COLOUR UP, AND COLOUR BACK DOWN. Tied to scroll rather than fired once, because
    // this is a figure and a figure re-derives itself every time the reader looks at it.
    //
    // It used to be the same fade the prose used, which is both the variety problem and a worse
    // reading of the number: pink squares fading up from the page's own ground read as squares
    // arriving out of nothing. Laid over a grey base they read as those squares TURNING, which is
    // what a refusal rate looks like when you watch it happen.
    //
    // The grid is handed over as the thing to watch. One animation across 175 elements gets one
    // observer, and left to itself it watches the first element, so the whole figure would have
    // been driven by the position of its top-left square.
    colourSweep(el, '.cell-refused');
  }, []);

  return (
    <div ref={root} aria-hidden="true" style={{ width: '100%' }}>
      <div className="grid-stack">
        {/* The grey bed: all four hundred, so there is a square under every square. */}
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: `repeat(${COLS}, 1fr)`,
            gap: '3px',
          }}
        >
          {Array.from({ length: TOTAL }, (_, i) => (
            // A class rather than an inline style on each of these nodes. The inline version put
            // the same five declarations into the HTML four hundred times and took the page from
            // 48KB to 104KB, on a page that makes a point of being small.
            <span key={i} className="cell-answered" />
          ))}
        </div>

        {/* The lit layer, exactly on top. The refusals are the first 175 cells in reading order,
            so this grid shares the bed's columns and gap and needs no spacers to line up. */}
        <div
          className="grid-lit"
          style={{
            display: 'grid',
            gridTemplateColumns: `repeat(${COLS}, 1fr)`,
            gap: '3px',
            alignContent: 'start',
          }}
        >
          {Array.from({ length: LIT }, (_, i) => (
            <span key={i} className="cell-refused" />
          ))}
        </div>
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
  // THE TEN REAL NUMBERS, in seed order 42 to 46, from
  // `evidence/k-sweep-2026-08-13/drift-per-seed.json`, which is committed in the repository and
  // cited by name in the paragraph under this chart. The means are not written down here, they
  // are computed from the seeds, so the bar cannot drift out of agreement with the ticks it is
  // drawn from. A test already recomputes every figure on this page from that same file.
  const BARS = [
    { label: 'One direction', k: 1, seeds: [0.0468, 0.0232, 0.0608, 0.07, 0.0477] },
    { label: 'Two directions', k: 2, seeds: [0.0622, 0.0719, 0.0708, 0.0828, 0.1784] },
  ].map((b) => {
    const mean = b.seeds.reduce((a, v) => a + v, 0) / b.seeds.length;
    // THE SPREAD COLUMN IS A SAMPLE STANDARD DEVIATION, not a range, and getting that wrong here
    // would have silently reprinted a different statistic under the same label. Checked against
    // the published figures: these five give 0.0177 and the other five give 0.0482, which is what
    // the table beside this chart says. The divisor is n-1.
    const variance =
      b.seeds.reduce((a, v) => a + (v - mean) ** 2, 0) / (b.seeds.length - 1);
    return { ...b, value: mean, spread: Math.sqrt(variance) };
  });
  // THE CEILING HAD TO RISE TO 0.19, and the reason is the whole point of the chart.
  // It was 0.14, which is comfortable headroom over the taller MEAN of 0.0932 and well under the
  // worst single run, 0.1784. Drawing the seeds with the old ceiling would have pushed that run
  // off the top of the frame, which is the one number a reader most needs to see: it is more than
  // double the next-worst run in its own arm and it is most of why our headline ratio is 1.9.
  // Hiding an outlier to keep the bars tall would be the chart lie this project exists to object
  // to. The ratio between the bars is unchanged by the ceiling.
  const MAX = 0.19;

  useEffect(() => {
    const el = root.current;
    if (!el) return;
    // THE SIGNATURE MOVE. Five hairlines land at the real heights of the five individual runs, in
    // seed order, and only then does the bar grow to the mean of them. Scroll back and the ticks
    // lift off and the bar retracts.
    //
    // It replaces two bars growing from the baseline, which was the same `grow` the corrections
    // spine uses and which told the reader to take the mean on trust. The order is the argument:
    // the evidence lands, then the summary is drawn out of it, and the reader watches one unlucky
    // run drag our own number up rather than reading about it in a spread column.
    seedAssembly(el);
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
          alignItems: 'stretch',
          gap: '1.4rem',
          height: '16rem',
          paddingBottom: '0.6rem',
          borderBottom: '1px solid var(--rule)',
        }}
      >
        {/* WHICH WAY IS GOOD, DRAWN RATHER THAN ONLY WRITTEN. The sentence above says lower is
            better; this says it again in geometry, because a reader who skipped the sentence is
            exactly the reader looking at the bars. The axis fades from nothing at the top to the
            accent at the bottom and ends in an arrowhead, so the good end of the scale is the
            coloured end. No words, because the words are three lines up. */}
        <div
          style={{
            width: '9px',
            position: 'relative',
            flex: '0 0 auto',
            background:
              'linear-gradient(to bottom, transparent 0%, var(--rule) 45%, var(--measured) 100%)',
            backgroundSize: '1px 100%',
            backgroundRepeat: 'no-repeat',
            backgroundPositionX: '4px',
          }}
        >
          <span
            style={{
              position: 'absolute',
              left: 0,
              bottom: 0,
              width: 0,
              height: 0,
              borderLeft: '4.5px solid transparent',
              borderRight: '4.5px solid transparent',
              borderTop: '7px solid var(--measured)',
            }}
          />
        </div>

        {BARS.map((b, i) => (
          <div key={b.label} style={{ flex: 1, textAlign: 'center', position: 'relative' }}>
            {/* THE COLUMN THE BAR, ITS SEEDS AND ITS LABEL ALL SHARE. The bar is anchored to the
                baseline and the ticks are placed by their own real heights against the same
                ceiling, so a tick and the bar it is averaged into are measured on one scale by
                construction rather than by two numbers that have to agree. */}
            <div style={{ position: 'absolute', inset: 0 }}>
              {/* THE BAR GOES IN FIRST AND THE TICKS GO OVER IT, which is the opposite of the
                  obvious order and the whole legibility of the figure depends on it. With the
                  ticks underneath, the bar swallowed four of its own five runs the moment it
                  finished growing and the only one left visible was the outlier, which then read
                  as a stray rule across the chart rather than as a measurement. Seen in a
                  screenshot; no amount of reading the markup would have shown it. */}
              <div
                data-bar
                style={{
                  position: 'absolute',
                  left: 0,
                  right: 0,
                  bottom: 0,
                  height: `${(b.value / MAX) * 100}%`,
                  minHeight: '4px',
                  borderRadius: '5px 5px 0 0',
                  transformOrigin: 'bottom',
                  opacity: 0.9,
                  background: i
                    ? 'linear-gradient(to bottom, #F98DB0, #8B4791)'
                    : 'var(--ink-faint)',
                }}
              />
              {b.seeds.map((s, n) => (
                <span
                  key={n}
                  data-seed
                  className={`seed-tick${i ? ' seed-tick--hi' : ''}`}
                  style={{ bottom: `${(s / MAX) * 100}%` }}
                />
              ))}
            </div>

            {/* The figure and its direction count ride just above the bar's own top edge rather
                than sitting at the top of the column. A number parked a hundred and fifty pixels
                above the thing it measures reads as a column heading, and on a two-column chart
                that is exactly the ambiguity the dots are here to remove. */}
            <div
              style={{
                position: 'absolute',
                left: 0,
                right: 0,
                bottom: `calc(${(b.value / MAX) * 100}% + 0.45rem)`,
              }}
            >
              {/* ONE SQUARE OR TWO, which is the fastest possible answer to "which bar is which".
                  The written labels under the chart say it too, and they are below the reader's
                  attention once the bars are moving. */}
              <div
                style={{
                  display: 'flex',
                  gap: '3px',
                  justifyContent: 'center',
                  marginBottom: '0.3rem',
                }}
              >
                {Array.from({ length: b.k }, (_, d) => (
                  <span
                    key={d}
                    style={{
                      width: '6px',
                      height: '6px',
                      borderRadius: '1px',
                      background: i ? 'var(--measured)' : 'var(--ink-faint)',
                    }}
                  />
                ))}
              </div>
              <p
                className="mono"
                style={{ margin: 0, fontSize: '0.95rem', color: 'var(--ink)' }}
              >
                {/* A ground behind the figure, because it now sits close enough to the bar to
                    land on one of the seed hairlines. Seen in a screenshot: 0.0496 printed
                    straight through the 0.0700 run. The chip is the page's own ground colour, so
                    it reads as the number interrupting the tick rather than as a label. */}
                <span
                  data-count={b.value}
                  data-decimals="4"
                  style={{ background: 'var(--bg)', padding: '0 0.35rem' }}
                >
                  {b.value.toFixed(4)}
                </span>
              </p>
            </div>
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: '1.4rem', marginTop: '0.7rem' }}>
        {/* A spacer the exact width of the axis column above, so each label still sits under its
            own bar. Without it every label is nine pixels and one gap to the left of the thing it
            names, which on two columns is enough to make the chart read as mislabelled. */}
        <div aria-hidden="true" style={{ width: '9px', flex: '0 0 auto' }} />
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
        Each hairline is one of the five seed runs the bar above it averages, at its own measured
        height.
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
