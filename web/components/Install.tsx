// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

'use client';

import { useEffect, useRef, useState } from 'react';
import { RunOutput } from '@/components/SectionArt';
import { onFirstView, reveal } from '@/lib/motion';

const DOCS = 'https://elementmerc.github.io/senbonzakura';
const NOTEBOOK =
  'https://colab.research.google.com/github/elementmerc/senbonzakura/blob/v0.4.1/notebooks/senbonzakura_colab.ipynb';

/**
 * NO TIMING CLAIM APPEARS HERE, and its absence is deliberate rather than an omission.
 *
 * Nobody has measured how long the notebook takes on Colab's hardware. The README says so in as
 * many words, and the obvious marketing line ("measure a model in five minutes") would be a
 * number nobody has. The one timing figure this project does have is the opposite of a selling
 * point: 14.3 minutes is the measured floor for the CPU demo on 16 cores, so a "three minute"
 * claim was dropped when somebody actually timed it.
 */
export default function Install() {
  const section = useRef<HTMLElement>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const el = section.current;
    if (!el) return;
    return onFirstView(el, () => reveal('[data-enter="install"]'));
  }, []);

  async function copy() {
    try {
      await navigator.clipboard.writeText('pip install senbonzakura');
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      // A denied or absent clipboard is not an error worth showing: the command is on screen and
      // selectable, which is how it was copied before this button existed.
    }
  }

  return (
    <>
      <hr className="rule" />
      <section ref={section} className="band">
        <div className="split">
        <div>
        <h2
          data-enter="install"
          style={{
            fontSize: 'var(--t-head)',
            lineHeight: 1.1,
            letterSpacing: '-0.025em',
            fontWeight: 600,
            margin: '0 0 1.1rem',
          }}
        >
          Run it on your own machine.
        </h2>

        <p
          data-enter="install"
          style={{ color: 'var(--ink-dim)', maxWidth: 'var(--measure)', margin: '0 0 2.2rem' }}
        >
          Python 3.10 or later. It works on a CPU, and a card makes it faster. Nothing you measure
          leaves the machine you measure it on.
        </p>

        <div
          data-enter="install"
          style={{ display: 'flex', flexWrap: 'wrap', gap: '0.8rem', alignItems: 'center' }}
        >
          <button
            type="button"
            onClick={copy}
            className="mono"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.8rem',
              background: 'var(--bg-raised)',
              border: '1px solid var(--rule)',
              borderRadius: '8px',
              padding: '0.9rem 1.2rem',
              fontSize: '0.95rem',
              color: 'var(--ink)',
              cursor: 'pointer',
            }}
          >
            <span>pip install senbonzakura</span>
            <span
              aria-live="polite"
              style={{ color: copied ? 'var(--sakura)' : 'var(--ink-faint)', fontSize: '0.8rem' }}
            >
              {copied ? 'copied' : 'copy'}
            </span>
          </button>

          <a
            href={NOTEBOOK}
            style={{
              padding: '0.9rem 1.2rem',
              fontSize: '0.95rem',
              textDecoration: 'none',
              border: '1px solid var(--rule)',
              borderRadius: '8px',
              color: 'var(--ink-dim)',
            }}
          >
            Open the notebook in Colab
          </a>
        </div>

        <p
          data-enter="install"
          style={{
            marginTop: '1.6rem',
            fontSize: 'var(--t-small)',
            color: 'var(--ink-faint)',
            maxWidth: 'var(--measure)',
            lineHeight: 1.7,
          }}
        >
          Free GPU, nothing on your machine. Nobody has timed it on Colab's hardware, so it comes
          with no duration to hold us to. The{' '}
          <a href={`${DOCS}/guide/first-run`}>first run guide</a> walks through what you get back,
          and <code>senbonzakura -i</code> asks the questions for you if you would rather not read
          the flags.
        </p>
        </div>
        <div data-enter="install" className="split-art">
          <RunOutput />
        </div>
        </div>
      </section>
    </>
  );
}
