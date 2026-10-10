// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

'use client';

import { useEffect, useRef } from 'react';
import { DriftBars } from '@/components/SectionArt';
import { countUp, reveal } from '@/lib/motion';

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

export default function Measured() {
  const section = useRef<HTMLElement>(null);

  useEffect(() => {
    const el = section.current;
    if (!el) return;
    reveal('[data-enter="measured"]');
    el.querySelectorAll<HTMLElement>('[data-count]').forEach((node) => {
      const to = Number(node.dataset.count);
      const decimals = Number(node.dataset.decimals ?? 0);
      if (Number.isFinite(to)) countUp(node, to, { decimals });
    });
  }, []);

  return (
    <section ref={section} className="band" style={{ paddingTop: 0 }}>
      <p
        data-enter="measured"
        className="mono"
        style={{
          color: 'var(--sakura)',
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
              <tr key={r.k} style={{ borderTop: i ? '1px solid var(--rule)' : undefined }}>
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
              color: 'var(--sakura)',
              margin: '0 0 0.6rem',
            }}
          >
            <span data-count="1.9" data-decimals="1">
              1.9
            </span>
            &times;
          </p>
          <p style={{ margin: '0 0 0.4rem', fontWeight: 500 }}>the collateral damage</p>
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
