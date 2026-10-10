// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

'use client';

import { useEffect, useRef } from 'react';
import { DriftBars } from '@/components/SectionArt';
import { countUp, intervalMarker, revealOnce, wipeInOnce } from '@/lib/motion';

/**
 * Our own figures, and the headline one is a result against our own flagship feature.
 *
 * Every number here is from the run of 2026-08-13: Qwen3-1.7B, one direction against two, same
 * tool, same corpus, same search budget, five seeds a side, then scored on 200 held-out prompts
 * that nothing was fitted or selected on.
 *
 * WHAT IS DELIBERATELY NOT HERE. That table also carried a noncompliance column, and it was
 * withdrawn on 2026-09-09 when the hedging detector was found to count statements of fact about
 * legality as hedging. It stays struck through on the docs page because it was published, and it
 * does not appear on this page at all, because reprinting a withdrawn figure in marketing would
 * be the exact failure the page is claiming not to have.
 *
 * The 1.9 is a ratio of means over five seeds a side and the interval is a percentile bootstrap.
 * It is printed WITH its interval because the docs page records that giving this figure as a bare
 * range once invited readers to mistake a sensitivity check for an interval, and the real one is
 * four times wider.
 */

const DOCS = 'https://elementmerc.github.io/senbonzakura';
const EVIDENCE =
  'https://github.com/elementmerc/senbonzakura/blob/v0.4.1/evidence/k-sweep-2026-08-13/drift-per-seed.json';

/**
 * THIS IS THE PAGE'S PEAK, and the motion is budgeted accordingly.
 *
 * People remember one moment and the ending. The middle compresses into a general impression and
 * then goes, so one act gets the asset budget, the silence in front of it, and the most scroll
 * room, at the expense of the others. The moment is the sentence below: our own flagship feature,
 * tested by us, losing.
 *
 * WHAT CHANGED HERE AND WHY. This section used to begin at `paddingTop: 0`, which gave the peak
 * the LEAST room on the page rather than the most, and it ran the same fade as every other
 * section. Now: an authored run of empty scroll in front of it so the reversal has something to
 * arrive out of, the table wiping in a row at a time rather than fading, the 95% interval drawn
 * under the headline ratio with its marker walking to where the measurement landed, and the chart
 * beside it assembling out of its own five runs.
 *
 * The empty run is deliberate and is recorded in the build brief. A verification pass that reports
 * dead scroll here is reporting the thing that was asked for.
 */
export default function Measured() {
  const section = useRef<HTMLElement>(null);
  const rail = useRef<HTMLDivElement>(null);
  const dot = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const el = section.current;
    if (!el) return;
    revealOnce('[data-enter="measured"]');
    // A WIPE, NOT A FADE, and once rather than both ways. A wipe reads as a change of state, which
    // is what a result arriving is, where a fade reads as something still loading. It holds after
    // it runs because a table is something a reader reads, and text that un-reveals when somebody
    // scrolls back up to check a figure is a defect wearing an effect's clothes.
    wipeInOnce('[data-row]', { step: 140 });
    el.querySelectorAll<HTMLElement>('[data-count]').forEach((node) => {
      const to = Number(node.dataset.count);
      const decimals = Number(node.dataset.decimals ?? 0);
      if (Number.isFinite(to)) countUp(node, to, { decimals });
    });

    if (!rail.current || !dot.current) return;
    // 1.2 and 3.1 are the published interval and 1.9 is the published ratio. Drawing them rather
    // than only printing them is the honest reading: the figure a reader repeats is the mean, and
    // the thing they should carry is how wide it is.
    return intervalMarker(dot.current, rail.current, { low: 1.2, high: 3.1, value: 1.9 });
  }, []);

  return (
    <section
      ref={section}
      className="band"
      style={{
        // THE PEAK'S SCROLL ROOM, and the one piece of spacing this rework changed. It was 0,
        // which made the loudest act on the page the one with the least air in front of it. The
        // empty run between the citations above and the first word here is the quiet the reversal
        // arrives out of; without it the peak follows a loud act directly and reads as more of the
        // same. feel.md calls this authored silence and it is in the brief as such.
        paddingTop: 'calc(var(--band) * 1.9)',
      }}
    >
      <p
        data-enter="measured"
        className="mono"
        style={{
          color: 'var(--measured)',
          fontSize: 'var(--t-small)',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          margin: '0 0 1.1rem',
        }}
      >
        Measured, not claimed
      </p>

      <h2
        data-enter="measured"
        style={{
          fontSize: 'var(--t-head)',
          lineHeight: 1.1,
          letterSpacing: '-0.025em',
          fontWeight: 600,
          margin: '0 0 1.1rem',
          maxWidth: '28ch',
        }}
      >
        We tested our own headline feature. It lost.
      </h2>

      <p
        data-enter="measured"
        style={{ color: 'var(--ink-dim)', maxWidth: 'var(--measure)', margin: '0 0 3rem' }}
      >
        Removing two refusal directions instead of one was the thing this tool was built to do.
        One direction against two, same corpus, same budget, five seeds a side, scored on 200
        prompts nothing was fitted on. Both left the model answering. Here is what the second
        direction cost.
      </p>

      <div className="split">
      <div
        data-enter="measured"
        style={{ border: '1px solid var(--rule)', borderRadius: '12px', overflow: 'hidden' }}
      >
        <table className="mono" style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.92rem' }}>
          <thead>
            <tr style={{ background: 'var(--bg-raised)' }}>
              <th style={{ textAlign: 'left', padding: '0.8rem 1.1rem', fontWeight: 500 }}>
                Directions removed
              </th>
              <th style={{ textAlign: 'right', padding: '0.8rem 1.1rem', fontWeight: 500 }}>
                Coherence drift
              </th>
              <th style={{ textAlign: 'right', padding: '0.8rem 1.1rem', fontWeight: 500 }}>
                Seed spread
              </th>
              <th style={{ textAlign: 'right', padding: '0.8rem 1.1rem', fontWeight: 500 }}>
                Hard refusal
              </th>
            </tr>
          </thead>
          <tbody>
            {[
              { k: 'One', drift: '0.0497', spread: '0.0177', refusal: '0.1%' },
              { k: 'Two', drift: '0.0932', spread: '0.0482', refusal: '0.3%' },
            ].map((r, i) => (
              <tr
                key={r.k}
                data-row
                style={{ borderTop: i ? '1px solid var(--rule)' : undefined }}
              >
                <td style={{ padding: '0.8rem 1.1rem' }}>{r.k}</td>
                <td style={{ padding: '0.8rem 1.1rem', textAlign: 'right' }}>{r.drift}</td>
                <td
                  style={{ padding: '0.8rem 1.1rem', textAlign: 'right', color: 'var(--ink-faint)' }}
                >
                  {r.spread}
                </td>
                <td style={{ padding: '0.8rem 1.1rem', textAlign: 'right' }}>{r.refusal}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div data-enter="measured" className="split-art">
        <DriftBars />
      </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(14rem, 1fr))',
          gap: '2.4rem 2rem',
          marginTop: '3rem',
          maxWidth: '46rem',
        }}
      >
        <div data-enter="measured">
          <p
            className="mono"
            style={{
              fontSize: 'clamp(2.2rem, 4vw, 3rem)',
              lineHeight: 1,
              letterSpacing: '-0.04em',
              color: 'var(--measured)',
              margin: '0 0 0.6rem',
            }}
          >
            <span data-count="1.9" data-decimals="1">
              1.9
            </span>
            &times;
          </p>
          <p style={{ margin: '0 0 0.4rem', fontWeight: 500 }}>the collateral damage</p>
          {/* THE INTERVAL, DRAWN. Every number in it is already printed in the note below, so this
              adds no claim; it turns one into geometry. The band is 1.2 to 3.1 at true position
              and the marker walks in from the low end to 1.9 as the reader arrives, and back out
              when they leave. `aria-hidden` because the sentence under it says the same thing in
              words, and a screen reader announcing a rule and a dot is worse than silence. */}
          <div ref={rail} className="interval" aria-hidden="true">
            {/* The rail IS the interval, so the band spans it end to end. */}
            <span className="interval-band" style={{ inset: '-3px 0 auto' }} />
            <span ref={dot} className="interval-dot" />
          </div>
          <p style={{ margin: 0, fontSize: 'var(--t-small)', color: 'var(--ink-faint)', lineHeight: 1.5 }}>
            95% interval 1.2 to 3.1, a percentile bootstrap over the five seeds a side. Printed
            with its interval because a bare range once read as one.
          </p>
        </div>

        <div data-enter="measured">
          <p
            className="mono"
            style={{
              fontSize: 'clamp(2.2rem, 4vw, 3rem)',
              lineHeight: 1,
              letterSpacing: '-0.04em',
              color: 'var(--ink)',
              margin: '0 0 0.6rem',
            }}
          >
            0
          </p>
          <p style={{ margin: '0 0 0.4rem', fontWeight: 500 }}>benefit this design could detect</p>
          <p style={{ margin: 0, fontSize: 'var(--t-small)', color: 'var(--ink-faint)', lineHeight: 1.5 }}>
            Both arms sit on the floor of the refusal measurement, so a gain had nowhere to show.
            That is a limit of the experiment, stated rather than read as a result.
          </p>
        </div>
      </div>

      <p
        data-enter="measured"
        style={{
          marginTop: '2.6rem',
          fontSize: 'var(--t-small)',
          color: 'var(--ink-faint)',
          lineHeight: 1.7,
          maxWidth: 'var(--measure)',
        }}
      >
        Qwen3-1.7B. One model, so refusal geometry may differ on larger ones. The ten per-seed
        values are committed at{' '}
        <a href={EVIDENCE} className="mono">
          evidence/k-sweep-2026-08-13/drift-per-seed.json
        </a>{' '}
        and a test recomputes every figure on the page from that file. The full working, including
        a column of this table we later withdrew, is in{' '}
        <a href={`${DOCS}/guide/what-we-know`}>what is and is not established</a>.
      </p>
    </section>
  );
}
