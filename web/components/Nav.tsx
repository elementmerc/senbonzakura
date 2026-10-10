// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

'use client';

import { useEffect, useState } from 'react';
import Mark from '@/components/Mark';

const DOCS = 'https://elementmerc.github.io/senbonzakura';

/**
 * The navigation, kept to what this project actually has.
 *
 * The reference site runs mega-menus over Models, Leaderboard, Offers, Playground, Developers and
 * Solutions, because it is a marketplace with that many destinations. We have documentation, the
 * evidence, and a repository. A nav padded out to look as busy as theirs would be four links that
 * go to the same docs site under different names, which a reader notices immediately and which
 * costs more trust than an empty nav bar ever would.
 *
 * It becomes solid on scroll rather than sitting on a permanent slab, so the hero reads as one
 * surface at rest.
 */
export default function Nav() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <nav
      style={{
        position: 'sticky',
        top: 0,
        zIndex: 20,
        borderBottom: `1px solid ${scrolled ? 'var(--rule)' : 'transparent'}`,
        background: scrolled ? 'rgba(14, 19, 48, 0.82)' : 'transparent',
        backdropFilter: scrolled ? 'blur(12px)' : 'none',
        transition: 'background 220ms ease, border-color 220ms ease',
      }}
    >
      <div
        style={{
          maxWidth: '76rem',
          margin: '0 auto',
          padding: '0.85rem var(--gutter)',
          display: 'flex',
          alignItems: 'center',
          gap: '1.4rem',
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

        {/* Hidden below a phone's width rather than folded into a burger menu. Three links do not
            earn a menu, and a burger holding three links is a tap a reader did not need. */}
        <div className="nav-links" style={{ display: 'flex', gap: '1.4rem', fontSize: '0.92rem' }}>
          <a href={`${DOCS}/guide/what-it-is`} style={{ textDecoration: 'none', color: 'var(--ink-dim)' }}>
            Docs
          </a>
          <a href={`${DOCS}/guide/what-we-know`} style={{ textDecoration: 'none', color: 'var(--ink-dim)' }}>
            Evidence
          </a>
          <a
            href={`${DOCS}/guide/what-we-got-wrong`}
            style={{ textDecoration: 'none', color: 'var(--ink-dim)' }}
          >
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

      <style>{`
        @media (max-width: 720px) {
          .nav-links { display: none !important; }
        }
      `}</style>
    </nav>
  );
}
