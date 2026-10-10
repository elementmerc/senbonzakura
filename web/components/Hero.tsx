// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

'use client';

import { useEffect } from 'react';
import { reveal } from '@/lib/motion';

/**
 * The headline is the project's own agreed sentence, shortened to fit a hero.
 *
 * The long form, from the positioning notes, is: "We do not abliterate better than Heretic. We
 * measure whether an abliteration worked, and nobody else in this field checks whether their own
 * measurement is any good." It leads with what we do NOT claim on purpose, because a technical
 * reader who arrives claiming-first then finds our own comparison page conceding two rows stops
 * believing the rest of it.
 *
 * The second line is the promise that no hosted competitor can answer, and it is load-bearing with
 * this audience rather than a feature bullet.
 */
export default function Hero() {
  useEffect(() => {
    document.documentElement.classList.add('js-ready');
    reveal('[data-enter="hero"]', { delay: 120 });
  }, []);

  return (
    <header className="band" style={{ paddingTop: 'clamp(4rem, 14vh, 9rem)' }}>
      <p
        data-enter="hero"
        className="mono"
        style={{
          color: 'var(--ink-faint)',
          fontSize: 'var(--t-small)',
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
          margin: '0 0 2.2rem',
        }}
      >
        Senbonzakura
      </p>

      <h1
        data-enter="hero"
        style={{
          fontSize: 'var(--t-hero)',
          lineHeight: 1.02,
          letterSpacing: '-0.035em',
          fontWeight: 560,
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
    </header>
  );
}
