// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

'use client';

import { animate, stagger, utils } from 'animejs';
import { useEffect, useRef } from 'react';
import { onFirstView, prefersReducedMotion } from '@/lib/motion';

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
    return onFirstView(el, () => {
      const cells = Array.from(el.querySelectorAll<HTMLElement>('.cell-refused'));
      if (prefersReducedMotion()) {
        utils.set(cells, { opacity: 1 });
        return;
      }
      animate(cells, {
        opacity: [0, 1],
        duration: 420,
        delay: stagger(4, { from: 'first' }),
        ease: 'outQuad',
      });
    });
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
    return onFirstView(el, () => {
      const bars = Array.from(el.querySelectorAll<HTMLElement>('[data-bar]'));
      if (prefersReducedMotion()) {
        bars.forEach((b) => {
          (b as HTMLElement).style.transform = 'scaleY(1)';
        });
        return;
      }
      animate(bars, {
        scaleY: [0, 1],
        duration: 1100,
        delay: stagger(140),
        ease: 'outExpo',
      });
    });
  }, []);

  return (
    <div ref={root} aria-hidden="true" style={{ width: '100%' }}>
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
              style={{ margin: '0 0 0.5rem', fontSize: '0.82rem', color: 'var(--ink)' }}
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
                transform: 'scaleY(0)',
                transformOrigin: 'bottom',
                background: i
                  ? 'linear-gradient(to bottom, #F98DB0, #8B4791)'
                  : 'var(--rule)',
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
              fontSize: '0.78rem',
              color: 'var(--ink-faint)',
              lineHeight: 1.5,
            }}
          >
            {b.label}
            <br />
            <span className="mono">spread {b.spread.toFixed(4)}</span>
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

  useEffect(() => {
    const el = root.current;
    if (!el) return;
    return onFirstView(el, () => {
      const rows = Array.from(el.querySelectorAll<HTMLElement>('[data-row]'));
      const still = prefersReducedMotion();
      animate(rows, {
        opacity: [0, 1],
        ...(still ? {} : { translateX: [-8, 0] }),
        duration: 500,
        delay: stagger(90),
        ease: 'outExpo',
      });
    });
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
            opacity: 0,
          }}
        >
          <span className="mono" style={{ fontSize: '0.82rem', color: 'var(--sakura)' }}>
            {name}
          </span>
          <span style={{ fontSize: '0.78rem', color: 'var(--ink-faint)' }}>{what}</span>
        </div>
      ))}
    </div>
  );
}
