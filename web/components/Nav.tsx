// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

'use client';

import { useEffect, useState } from 'react';
import Mark from '@/components/Mark';

const DOCS = 'https://elementmerc.github.io/senbonzakura';

/**
 * A full-width bar at rest that becomes a floating island once you scroll.
 *
 * THE BUG THIS REPLACES. The first version was transparent until `scrollY > 12` and took its
 * background from a state flag. Whenever that flag was wrong, or during the frame before it was
 * set, the page's own headings ran straight through the wordmark and the links: screenshots
 * showed "43.8%" printed across "SENBONZAKURA". A navigation bar that is sometimes invisible over
 * moving content is unreadable exactly when somebody is using it.
 *
 * So the background is no longer conditional. Both states are opaque enough to read against, and
 * the scroll flag now only chooses WHICH solid treatment applies rather than whether there is one.
 * A state flag that fails open to "no background" was the design error, not the flag.
 *
 * The island shape is the effect from abliteration.ai: flush and full width at the top, then
 * inset from the edges with a rounded border and a blur once the page moves. It reads as the bar
 * lifting off the page.
 *
 * The transform is marked `data-motion="travels"`, so a reader who has asked for less movement
 * gets both states and no sliding between them.
 */
export default function Nav() {
  const [lifted, setLifted] = useState(false);

  useEffect(() => {
    // rAF-throttled: a scroll handler that calls setState on every event re-renders this
    // component dozens of times a second for a boolean that changes twice.
    let frame = 0;
    const onScroll = () => {
      if (frame) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        setLifted(window.scrollY > 24);
      });
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', onScroll);
      if (frame) cancelAnimationFrame(frame);
    };
  }, []);

  return (
    <div
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 40,
        padding: lifted ? '0.7rem var(--gutter) 0' : '0',
        transition: 'padding 320ms var(--ease)',
      }}
      data-motion="travels"
    >
      <nav
        style={{
          maxWidth: lifted ? '72rem' : '100%',
          margin: '0 auto',
          borderRadius: lifted ? '999px' : '0',
          border: '1px solid',
          borderColor: lifted ? 'var(--rule)' : 'transparent',
          borderBottomColor: 'var(--rule)',
          // ALWAYS a background. This is the line the old version got wrong.
          background: lifted ? 'rgba(20, 26, 66, 0.72)' : 'var(--navy-deep)',
          backdropFilter: lifted ? 'blur(14px) saturate(140%)' : 'none',
          boxShadow: lifted ? '0 10px 30px rgba(5, 8, 24, 0.45)' : 'none',
          transition:
            'max-width 320ms var(--ease), border-radius 320ms var(--ease), background 320ms ease, box-shadow 320ms ease, border-color 320ms ease',
        }}
      >
        <div
          style={{
            maxWidth: '76rem',
            margin: '0 auto',
            padding: lifted ? '0.6rem 1.2rem' : '0.85rem var(--gutter)',
            display: 'flex',
            alignItems: 'center',
            gap: '1.4rem',
            transition: 'padding 320ms var(--ease)',
          }}
        >
          <a
            href="/"
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.55rem',
              textDecoration: 'none',
              marginRight: 'auto',
            }}
          >
            <Mark size={26} />
            <span className="wordmark" style={{ fontSize: '0.85rem' }}>
              Senbonzakura
            </span>
          </a>

          {/* Hidden below a phone's width rather than folded into a burger menu. Three links do
              not earn a menu, and a burger holding three links is a tap nobody needed. */}
          <div className="nav-links" style={{ display: 'flex', gap: '1.4rem', fontSize: '0.92rem' }}>
            <a href={`${DOCS}/guide/what-it-is`} style={{ textDecoration: 'none', color: 'var(--ink-dim)' }}>
              Docs
            </a>
            <a href="/corrections" style={{ textDecoration: 'none', color: 'var(--ink-dim)' }}>
              What we got wrong
            </a>
          </div>

          <a
            href="https://github.com/elementmerc/senbonzakura"
            style={{
              textDecoration: 'none',
              fontSize: '0.92rem',
              padding: '0.45rem 0.95rem',
              borderRadius: '999px',
              border: '1px solid var(--rule)',
              color: 'var(--ink)',
              whiteSpace: 'nowrap',
            }}
          >
            GitHub
          </a>
        </div>
      </nav>

      <style>{`
        @media (max-width: 760px) {
          .nav-links { display: none !important; }
        }
      `}</style>
    </div>
  );
}
