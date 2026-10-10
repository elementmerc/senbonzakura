// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

'use client';

import { useEffect, useRef } from 'react';
import { growIn, reveal } from '@/lib/motion';

/**
 * Twelve corrections as a descending timeline, alternating either side of a spine.
 *
 * WHY THERE ARE TWO DATES AND NOT TWELVE. Every one of these was published on 2026-09-27 or
 * 2026-10-01, which git confirms for each entry. A month-by-month timeline would therefore be
 * two rows pretending to be a history, so the dates group the entries rather than label each
 * one. Inventing a plausible date per row would have made a prettier graphic and a false one,
 * on the page whose entire subject is claims that turned out false.
 *
 * Each row is the two phrases the source page already uses: what it said, and what it says now.
 * The content is a summary of `docs/guide/what-we-got-wrong.md`, and the page itself remains the
 * long form; the link under each cluster goes there.
 */

const DOCS = 'https://elementmerc.github.io/senbonzakura';

type Correction = { was: string; now: string };
type Cluster = { date: string; label: string; items: Correction[] };

const CLUSTERS: Cluster[] = [
  {
    date: '2026-09-27',
    label: 'Eight claims about how to run it',
    items: [
      {
        was: 'Do not run pip install senbonzakura.',
        now: 'That is the whole install, and it always was.',
      },
      {
        was: 'The first run takes 108 minutes on Qwen3-1.7B.',
        now: 'The 108 minutes were measured on a different, smaller model.',
      },
      {
        was: 'Abliterate Qwen3-1.7B, on every page showing a first command.',
        now: 'Abliterate Qwen2.5-0.5B-Instruct. The old demo barely refused anything.',
      },
      {
        was: 'The whole edit fits in a three minute CPU demo.',
        now: '14.3 minutes on sixteen cores, with the budget cut to the bone.',
      },
      {
        was: 'Qwen3-4B will not run on a 6 GB card.',
        now: 'It loads. The overflow goes to host RAM and the run completes.',
      },
      {
        was: 'Sixteen flags, 55 more, and 69 in total.',
        now: 'All three counted from the parser when the help is built.',
      },
      {
        was: 'Three different install sizes, on three pages.',
        now: '70 packages and 5.9 GB, measured once in an empty environment.',
      },
      {
        was: 'Abliterated checkpoints are published on the Hub.',
        now: 'None were. Checked with no credentials, the way a stranger would.',
      },
    ],
  },
  {
    date: '2026-10-01',
    label: 'Four claims about what our numbers mean',
    items: [
      {
        was: 'Every one of the seven commands reports an interval.',
        now: 'Every measurement does, with the two exceptions named on the spot.',
      },
      {
        was: 'Everyone has the norm-restore problem.',
        now: 'Three tools do. That is what we measured, and the rest is unchecked.',
      },
      {
        was: 'Drift is how far the output distribution moved.',
        now: 'How far the first-token distribution moved, one position per prompt.',
      },
      {
        was: 'Nobody validates the judge.',
        now: 'Nobody computes agreement above chance. One rival does pin a judge.',
      },
    ],
  },
];

export default function Corrections() {
  const section = useRef<HTMLElement>(null);

  useEffect(() => {
    reveal('[data-enter="corr"]');
    // The spine draws downward, tied to scroll, so it retracts on the way back up. That is the
    // one piece of motion here that carries the graphic's meaning: a sequence, in order, going
    // down. `transformOrigin: top` is set in CSS, because growIn cannot know which end grows.
    growIn('[data-spine]', { axis: 'scaleY', duration: 1200, step: 0 });
  }, []);

  return (
    <>
      <hr className="rule" />
      <section ref={section} id="corrections" className="band">
        <p
          data-enter="corr"
          className="mono"
          style={{
            color: 'var(--sakura)',
            fontSize: 'var(--t-small)',
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            margin: '0 0 1.1rem',
          }}
        >
          What we got wrong
        </p>

        <h2
          data-enter="corr"
          style={{
            fontSize: 'var(--t-head)',
            lineHeight: 1.1,
            letterSpacing: '-0.025em',
            fontWeight: 600,
            margin: '0 0 1.1rem',
            maxWidth: '24ch',
          }}
        >
          Twelve things we published that were not true.
        </h2>

        <p
          data-enter="corr"
          style={{ color: 'var(--ink-dim)', maxWidth: 'var(--measure)', margin: '0 0 4rem' }}
        >
          They are still on the site, with the original wording beside the correction. Neither
          competitor publishes a page like this.
        </p>

        <div style={{ position: 'relative', maxWidth: '62rem', margin: '0 auto' }}>
          {/* The spine. `transformOrigin: top` so it grows downward rather than outward. */}
          <div
            data-spine
            data-motion="travels"
            aria-hidden="true"
            style={{
              position: 'absolute',
              left: 'var(--spine-x)',
              top: 0,
              bottom: 0,
              width: '2px',
              marginLeft: '-1px',
              background:
                'linear-gradient(to bottom, #F98DB0 0%, #E06A9C 38%, #8B4791 66%, #141A42 100%)',
              transformOrigin: 'top',
            }}
          />

          {CLUSTERS.map((cluster) => (
            <div key={cluster.date}>
              <div
                data-enter="corr"
                style={{
                  position: 'relative',
                  paddingLeft: 'var(--row-pad)',
                  marginBottom: '2.2rem',
                }}
              >
                <span
                  className="mono"
                  style={{
                    display: 'inline-block',
                    padding: '0.3rem 0.8rem',
                    borderRadius: '999px',
                    background: 'var(--bg-raised)',
                    border: '1px solid var(--rule)',
                    fontSize: '0.78rem',
                    color: 'var(--sakura)',
                  }}
                >
                  {cluster.date}
                </span>
                <span
                  style={{ marginLeft: '0.8rem', fontSize: '0.95rem', color: 'var(--ink-dim)' }}
                >
                  {cluster.label}
                </span>
              </div>

              {cluster.items.map((item, i) => (
                <div
                  key={item.was}
                  data-enter="corr"
                  className={`corr-row ${i % 2 ? 'corr-right' : 'corr-left'}`}
                >
                  {/* The node on the spine. */}
                  <span className="corr-node" aria-hidden="true" />
                  <p className="corr-was">{item.was}</p>
                  <p className="corr-now">{item.now}</p>
                </div>
              ))}
            </div>
          ))}
        </div>

        <p
          data-enter="corr"
          style={{
            marginTop: '3rem',
            fontSize: 'var(--t-small)',
            color: 'var(--ink-faint)',
            lineHeight: 1.7,
          }}
        >
          Both dates are real and there are only two of them, because that is when the corrections
          were published rather than when each mistake was made. Each one has its own{' '}
          <a href="/corrections">page here</a>, and the long form with the original wording and
          what was actually wrong is in{' '}
          <a href={`${DOCS}/guide/what-we-got-wrong`}>what we got wrong</a>.
        </p>

        <style>{`
          .corr-row {
            position: relative;
            margin-bottom: 2rem;
            padding-left: var(--row-pad);
          }
          .corr-node {
            position: absolute;
            left: var(--spine-x);
            top: 0.45rem;
            width: 9px;
            height: 9px;
            margin-left: -4.5px;
            border-radius: 50%;
            background: var(--sakura);
            box-shadow: 0 0 0 4px var(--bg);
          }
          .corr-was {
            margin: 0 0 0.35rem;
            color: var(--ink-faint);
            text-decoration: line-through;
            text-decoration-color: rgba(247, 121, 159, 0.5);
          }
          .corr-now { margin: 0; color: var(--ink); }

          /* ONE COLUMN ON A PHONE, two from 860px. The alternating layout needs room for two
             readable measures either side of the spine, and below that it becomes two narrow
             columns of broken words, which is the usual failure of this graphic. */
          :root { --spine-x: 7px; --row-pad: 2rem; }
          @media (min-width: 860px) {
            :root { --spine-x: 50%; --row-pad: 0; }
            .corr-row { width: 50%; padding-left: 0; }
            .corr-left { text-align: right; padding-right: 2.6rem; }
            .corr-right { margin-left: 50%; padding-left: 2.6rem; }
            .corr-left .corr-node { left: 100%; }
            .corr-right .corr-node { left: 0; }
          }
        `}</style>
      </section>
    </>
  );
}
