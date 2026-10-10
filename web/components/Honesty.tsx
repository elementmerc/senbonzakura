// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

'use client';

import { useEffect, useRef } from 'react';
import { reveal } from '@/lib/motion';

const DOCS = 'https://elementmerc.github.io/senbonzakura';

/**
 * The two pages nobody else in this field publishes, and one worked example of why they exist.
 *
 * Both competitors publish favourable figures and no corrections. The asset here is not that our
 * numbers are better, because on the comparison that matters they are not. It is that every
 * figure traces to a file and a command, and that the ones we got wrong are still on the site
 * with the correction beside them.
 *
 * The example is real and is the strongest one available: a column of our own published table,
 * struck through rather than deleted, because the detector behind it counted a factual statement
 * about legality as a refusal.
 */
export default function Honesty() {
  const section = useRef<HTMLElement>(null);

  useEffect(() => {
    reveal('[data-enter="honesty"]');
  }, []);

  return (
    <>
      <hr className="rule" />
      <section ref={section} className="band">
        <h2
          data-enter="honesty"
          style={{
            fontSize: 'var(--t-head)',
            lineHeight: 1.1,
            letterSpacing: '-0.025em',
            fontWeight: 600,
            margin: '0 0 1.1rem',
            maxWidth: '24ch',
          }}
        >
          Every figure traces to a file and a command.
        </h2>

        <p
          data-enter="honesty"
          style={{ color: 'var(--ink-dim)', maxWidth: 'var(--measure)', margin: '0 0 2.8rem' }}
        >
          Including the ones we got wrong, which stay published with the correction next to them.
        </p>

        <figure
          data-enter="honesty"
          style={{
            margin: '0 0 2.8rem',
            padding: '1.4rem 1.6rem',
            borderLeft: '2px solid var(--measured)',
            background: 'var(--bg-raised)',
            borderRadius: '0 10px 10px 0',
            maxWidth: '46rem',
          }}
        >
          <blockquote style={{ margin: 0, color: 'var(--ink-dim)', lineHeight: 1.7 }}>
            The hedging detector was found to count statements of fact about legality or danger as
            hedging, so a complete, compliant technical answer that mentioned an activity is
            illegal was scored as a soft refusal. The column is struck through rather than deleted
            because it was published.
          </blockquote>
          <figcaption
            style={{ marginTop: '0.9rem', fontSize: 'var(--t-small)', color: 'var(--ink-faint)' }}
          >
            From our own results page, 2026-09-09.
          </figcaption>
        </figure>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(17rem, 1fr))',
            gap: '1rem',
            maxWidth: '46rem',
          }}
        >
          {[
            {
              href: `${DOCS}/guide/what-we-know`,
              title: 'What is and is not established',
              body: 'Each claim, the run behind it, and where it stops being supported.',
            },
            {
              href: `${DOCS}/guide/what-we-got-wrong`,
              title: 'What we got wrong',
              body: 'The corrections, kept with the original wording so you can see what changed.',
            },
            {
              href: `${DOCS}/guide/limits`,
              title: 'Limits and known defects',
              body: 'What is still true and still broken, written down before you find it.',
            },
            {
              href: 'https://github.com/elementmerc/senbonzakura/blob/v0.4.1/REPRODUCING.md',
              title: 'Reproducing it',
              body: 'Every published figure mapped to the file it came from and the command that makes it.',
            },
          ].map((c) => (
            <a
              key={c.title}
              href={c.href}
              data-enter="honesty"
              style={{
                display: 'block',
                padding: '1.2rem 1.3rem',
                border: '1px solid var(--rule)',
                borderRadius: '10px',
                textDecoration: 'none',
              }}
            >
              <p style={{ margin: '0 0 0.4rem', fontWeight: 500 }}>{c.title}</p>
              <p
                style={{
                  margin: 0,
                  fontSize: 'var(--t-small)',
                  color: 'var(--ink-faint)',
                  lineHeight: 1.55,
                }}
              >
                {c.body}
              </p>
            </a>
          ))}
        </div>
      </section>
    </>
  );
}
