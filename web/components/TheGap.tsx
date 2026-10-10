// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

'use client';

import { useEffect, useRef } from 'react';
import { RefusalGrid } from '@/components/SectionArt';
import { countUp, onFirstView, reveal } from '@/lib/motion';

/**
 * NOT ONE OF THESE NUMBERS IS OURS, and that is the point of the block.
 *
 * A landing page quoting its own favourable figures asks to be taken on trust. These are other
 * people's measurements, both papers read first-hand and cited in full, so a sceptical reader can
 * go and check them without involving us. Each figure carries its source on the page rather than
 * in a footnote, because a number whose provenance is one click away is a number somebody will
 * repeat without the provenance.
 *
 * The last row is the argument. The Beyond Refusal authors close by calling for evaluations that
 * jointly measure refusal, correctness and actionability, and their own paper does not measure
 * what the edit cost. Somebody else named the hole in their own conclusion.
 */

type Figure = {
  value: number;
  decimals: number;
  suffix: string;
  label: string;
  note: string;
};

const FIGURES: Figure[] = [
  {
    value: 43.8,
    decimals: 1,
    suffix: '%',
    label: 'of system hardening requests refused',
    note: 'Measured on 2,390 conversations from a sanctioned collegiate blue-team competition.',
  },
  {
    value: 2.72,
    decimals: 2,
    suffix: '×',
    label: 'more refusals for attack-shaped wording',
    note: 'The same request, worded with exploit or payload or shell, against a neutral phrasing.',
  },
  {
    value: 21.8,
    decimals: 1,
    suffix: '%',
    label: 'refused when you say you are authorised',
    note: 'Against 11.6% when you say nothing, and 50.0% when authorisation and security wording appear together.',
  },
  {
    value: 67.8,
    decimals: 1,
    suffix: '%',
    label: 'usable vulnerability patches after an edit',
    note: 'Against 29.94% from the same weights before the edit, on Vul4J.',
  },
];

export default function TheGap() {
  const section = useRef<HTMLElement>(null);

  useEffect(() => {
    const el = section.current;
    if (!el) return;

    return onFirstView(el, () => {
      reveal('[data-enter="gap"]');
      el.querySelectorAll<HTMLElement>('[data-count]').forEach((node, i) => {
        const to = Number(node.dataset.count);
        const decimals = Number(node.dataset.decimals ?? 0);
        if (Number.isFinite(to)) countUp(node, to, { decimals, delay: 140 + i * 90 });
      });
    });
  }, []);

  return (
    <section ref={section} className="band">
        <h2
          data-enter="gap"
          style={{
            fontSize: 'var(--t-head)',
            lineHeight: 1.1,
            letterSpacing: '-0.025em',
            fontWeight: 560,
            margin: '0 0 1.1rem',
            maxWidth: '26ch',
          }}
        >
          A safety-tuned model refuses defenders.
        </h2>

        <p
          data-enter="gap"
          style={{
            color: 'var(--ink-dim)',
            maxWidth: 'var(--measure)',
            margin: '0 0 3.4rem',
          }}
        >
          None of these numbers is ours. Two research groups measured them, we read both papers,
          and you can check either one without asking us anything.
        </p>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(15rem, 1fr))',
            gap: '2.6rem 2rem',
          }}
        >
          {FIGURES.map((f) => (
            <div key={f.label} data-enter="gap">
              <p
                className="mono"
                style={{
                  fontSize: 'clamp(2.4rem, 4.4vw, 3.4rem)',
                  lineHeight: 1,
                  letterSpacing: '-0.04em',
                  color: 'var(--measured)',
                  margin: '0 0 0.7rem',
                }}
              >
                {/* The final value is the element's own content, so the figure is correct before
                    any animation runs and for a reader who never gets one. */}
                <span data-count={f.value} data-decimals={f.decimals}>
                  {f.value.toFixed(f.decimals)}
                </span>
                {f.suffix}
              </p>
              <p style={{ margin: '0 0 0.5rem', fontWeight: 500 }}>{f.label}</p>
              <p
                style={{
                  margin: 0,
                  fontSize: 'var(--t-small)',
                  color: 'var(--ink-faint)',
                  lineHeight: 1.5,
                }}
              >
                {f.note}
              </p>
            </div>
          ))}
        </div>

        <div
          className="split"
          style={{ marginTop: '3.4rem', paddingTop: '2rem', borderTop: '1px solid var(--rule)' }}
        >
          <div data-enter="gap" style={{ maxWidth: 'var(--measure)' }}>
          <p style={{ margin: '0 0 1rem' }}>
            So people edit the refusal out. The second paper above found that doubles the usable
            patches, then closed by asking for evaluations that measure refusal, correctness and
            actionability together.
          </p>
          <p style={{ margin: 0, color: 'var(--ink-dim)' }}>
            Its own paper does not measure what the edit cost. That is what this tool is for.
          </p>

          <p
            style={{
              marginTop: '2rem',
              marginBottom: 0,
              fontSize: 'var(--t-small)',
              color: 'var(--ink-faint)',
              lineHeight: 1.7,
            }}
          >
            Campbell et al.,{' '}
            <a href="https://arxiv.org/abs/2603.01246">
              Defensive Refusal Bias: How Safety Alignment Fails Cyber Defenders
            </a>
            , Security and Policy Research Lab, Scale AI, ICLR 2026 workshop.
            <br />
            Li et al.,{' '}
            <a href="https://arxiv.org/abs/2607.05842">
              Beyond Refusal: A Same-Lineage Study of Aligned and Abliterated LLMs for
              Vulnerability Analysis
            </a>
            , July 2026.
          </p>
          </div>
          <div data-enter="gap" className="split-art">
            <RefusalGrid />
          </div>
        </div>
    </section>
  );
}
