// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

'use client';

import { useEffect, useRef } from 'react';
import Mark from '@/components/Mark';
import { driftPetals, reveal } from '@/lib/motion';

/**
 * The headline is the project's own agreed sentence, shortened to fit a hero.
 *
 * The long form, from the positioning notes, is: "We do not abliterate better than Heretic. We
 * measure whether an abliteration worked, and nobody else in this field checks whether their own
 * measurement is any good." It leads with what we do NOT claim on purpose, because a technical
 * reader who arrives claiming-first then finds our own comparison page conceding two rows stops
 * believing the rest of it.
 *
 * The second line is the promise no hosted competitor can answer, and it is load-bearing with
 * this audience rather than a feature bullet.
 */
export default function Hero() {
  const root = useRef<HTMLElement>(null);

  useEffect(() => {
    document.documentElement.classList.add('js-ready');
    reveal('[data-enter="hero"]', { delay: 120 });
    const el = root.current;
    if (!el) return;
    return driftPetals(el);
  }, []);

  return (
    <header ref={root} className="band" style={{ paddingTop: 'clamp(3rem, 11vh, 7rem)' }}>
      {/* The lockup: mark beside wordmark. The kit ships this as an SVG, and it is set as live
          text here instead so it inherits the page's own type and stays crisp at any size. The
          kit's own rule for the wordmark is what is applied: uppercase, about 0.14em tracking. */}
      <div
        data-enter="hero"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.7rem',
          margin: '0 0 2.6rem',
        }}
      >
        <Mark size={40} />
        <span
          className="wordmark"
          style={{ fontSize: '1.02rem', color: 'var(--ink)' }}
        >
          Senbonzakura
        </span>
        <span
          className="jp"
          aria-hidden="true"
          style={{ color: 'var(--ink-faint)', fontSize: '1rem', marginLeft: '0.1rem' }}
        >
          千本桜
        </span>
      </div>

      <h1
        data-enter="hero"
        style={{
          fontSize: 'var(--t-hero)',
          lineHeight: 1.02,
          letterSpacing: '-0.035em',
          fontWeight: 600,
          margin: '0 0 1.6rem',
          maxWidth: '22ch',
        }}
      >
        The refusal fell.
        <br />
        <span style={{ color: 'var(--ink-dim)' }}>What else moved?</span>
      </h1>

      <p
        data-enter="hero"
        style={{
          fontSize: 'clamp(1.05rem, 1.5vw, 1.3rem)',
          color: 'var(--ink-dim)',
          maxWidth: 'var(--measure)',
          margin: '0 0 2.6rem',
        }}
      >
        Senbonzakura measures what a behaviour edit did to an open-weight model, and what it cost.
        It runs on your machine and makes no network call you did not ask for.
      </p>

      <div
        data-enter="hero"
        style={{ display: 'flex', flexWrap: 'wrap', gap: '0.8rem', alignItems: 'center' }}
      >
        <code
          style={{
            background: 'var(--bg-raised)',
            border: '1px solid var(--rule)',
            borderRadius: '7px',
            padding: '0.8rem 1.1rem',
            fontSize: '0.95rem',
          }}
        >
          pip install senbonzakura
        </code>
        <a
          href="https://elementmerc.github.io/senbonzakura/guide/quickstart"
          style={{
            padding: '0.8rem 1.1rem',
            fontSize: '0.95rem',
            textDecoration: 'none',
            border: '1px solid var(--rule)',
            borderRadius: '7px',
            color: 'var(--ink-dim)',
          }}
        >
          Read the docs
        </a>
      </div>

      {/* Facts, not a feature list. Each one is checkable by somebody who does not trust us,
          which is the only kind of claim worth putting under a hero. */}
      <p
        data-enter="hero"
        className="mono"
        style={{
          color: 'var(--ink-faint)',
          fontSize: 'var(--t-small)',
          marginTop: '1.6rem',
        }}
      >
        0.4.1 on PyPI &nbsp;·&nbsp; AGPL-3.0-or-later &nbsp;·&nbsp; runs on CPU &nbsp;·&nbsp; free
        GPU notebook
      </p>

      {/* THE ONE PLACE THE GRADIENT APPEARS AS A GRADIENT, as a hairline closing the hero. The
          kit insists it runs pink to violet to navy and that the violet midpoint is load-bearing,
          so it is rotated to a horizontal sweep rather than recoloured or reordered. */}
      <div
        data-enter="hero"
        aria-hidden="true"
        style={{
          marginTop: 'calc(var(--band) * 0.55)',
          height: '2px',
          background: 'linear-gradient(to right, #F98DB0 0%, #E06A9C 38%, #8B4791 66%, #141A42 100%)',
          borderRadius: '2px',
        }}
      />
    </header>
  );
}
