// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

'use client';

import { useEffect, useRef } from 'react';
import Mark from '@/components/Mark';
import { driftPetals, magnet, pointerWash, revealOnce, riseLines } from '@/lib/motion';

/**
 * The headline is the project's own agreed sentence, shortened to fit a hero.
 *
 * The long form, from the positioning notes, is: "We do not abliterate better than Heretic. We
 * measure whether an abliteration worked, and nobody else in this field checks whether their own
 * measurement is any good." It leads with what we do NOT claim on purpose, because a technical
 * reader who arrives claiming-first then finds our own comparison page conceding two rows stops
 * believing the rest of it.
 *
 * The second line used to read "no network call you did not ask for". WITHDRAWN 2026-10-10:
 * decision Q-94 puts opt-out telemetry in the tool behind a first-run notice, and some features
 * are online by design, so the old line claimed something the product does not keep.
 *
 * What replaces it is narrower, true, and still something a hosted service structurally cannot
 * say: the weights stay put. A hosted competitor RUNS the model, so it holds your weights by
 * definition. Q-94's own hard line is the same shape, an architecture fingerprint and never a
 * model name, because for a lab measuring an unreleased model the name is the leak.
 */
/**
 * THE ACT'S DEVICE IS KINETIC TYPE, and it is the only act on the page that uses it.
 *
 * The headline's two lines rise out from behind masks, once, on load, and then the hero holds
 * still. It used to fade and rise like everything else on the page, which is the right fade and
 * the wrong thing to do five times; and because that fade was tied to scroll it also meant the
 * hero UN-REVEALED itself whenever somebody scrolled back to the top. Measured: every headline
 * element at opacity 0 on returning to y=0.
 *
 * What the hero does after that is respond to the reader rather than perform at them. A wash of
 * the accent follows the pointer behind the type, and the pill leans toward it. Both are gated to
 * a real pointer, so a phone gets a hero that is simply still, which is the correct hero for a
 * page whose first claim is about restraint.
 */
export default function Hero() {
  const root = useRef<HTMLElement>(null);
  const pill = useRef<HTMLAnchorElement>(null);

  useEffect(() => {
    document.documentElement.classList.add('js-ready');
    riseLines('.kline > span');
    revealOnce('[data-enter="hero"]');

    const el = root.current;
    if (!el) return;
    const stop = [driftPetals(el), pointerWash(el)];
    if (pill.current) stop.push(magnet(pill.current));
    return () => stop.forEach((fn) => fn());
  }, []);

  return (
    <header
      ref={root}
      className="band"
      style={{
        paddingTop: 'clamp(2.5rem, 9vh, 6rem)',
        textAlign: 'center',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        // The wash is a child at a negative z-index, so the header has to be the thing it is
        // behind. `isolation` keeps it inside this stacking context rather than letting it slide
        // under the page's own ground, where it would be invisible.
        position: 'relative',
        isolation: 'isolate',
      }}
    >
      <div className="hero-wash" aria-hidden="true" />

      {/* A pill above the headline, which is where the reference site puts the one claim it
          most wants read. Ours is the promise a hosted competitor structurally cannot make.

          NO `data-enter` HERE, deliberately. The magnet writes `transform` every frame in its own
          loop, so a reveal on the same element loses its rise to a race and the reader sees
          nothing. One continuous transform per element. */}
      <a
        ref={pill}
        href="#install"
        style={{
          display: 'inline-flex',
          alignItems: 'center',
          gap: '0.6rem',
          padding: '0.5rem 1rem 0.5rem 0.75rem',
          marginBottom: '2.2rem',
          border: '1px solid var(--rule)',
          borderRadius: '999px',
          background: 'var(--bg-raised)',
          fontSize: '0.88rem',
          textDecoration: 'none',
          color: 'var(--ink-dim)',
        }}
      >
        <Mark size={18} />
        <span>Your weights never leave your machine.</span>
        <span aria-hidden="true" style={{ color: 'var(--ink-faint)' }}>
          &rarr;
        </span>
      </a>

      {/* TWO MASKS, WRITTEN OUT RATHER THAN MEASURED. The usual implementation wraps every word in
          a span, reads the real line boxes and re-runs after the webfont lands. On a headline that
          carries a hard break and a coloured second half that rebuild has to reconstruct the
          colour, and a measurement taken before the font arrives groups the words wrongly. This
          headline has exactly two lines by authorial decision, so there is nothing to measure and
          no font-loading order to get right. The `<br/>` is gone because each line is its own
          block now; the words are unchanged. */}
      <h1
        style={{
          fontSize: 'var(--t-hero)',
          lineHeight: 1.02,
          letterSpacing: '-0.035em',
          fontWeight: 600,
          margin: '0 0 1.6rem',
          maxWidth: '20ch',
        }}
      >
        <span className="kline">
          <span>The refusal fell.</span>
        </span>
        <span className="kline">
          <span style={{ color: 'var(--ink-dim)' }}>What else moved?</span>
        </span>
      </h1>

      <p
        data-enter="hero"
        style={{
          fontSize: 'clamp(1.05rem, 1.5vw, 1.3rem)',
          color: 'var(--ink-dim)',
          maxWidth: '54ch',
          margin: '0 0 2.6rem',
        }}
      >
        Senbonzakura measures what a behaviour edit did to an open-weight model, and what it cost.
        The measuring runs on your own hardware.
      </p>

      <div
        data-enter="hero"
        style={{ display: 'flex', flexWrap: 'wrap', gap: '0.8rem', alignItems: 'center', justifyContent: 'center' }}
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
          width: 'min(100%, 46rem)',
          height: '2px',
          background: 'linear-gradient(to right, #F98DB0 0%, #E06A9C 38%, #8B4791 66%, #141A42 100%)',
          borderRadius: '2px',
        }}
      />
    </header>
  );
}
