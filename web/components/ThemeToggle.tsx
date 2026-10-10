// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

'use client';

import { useEffect, useState } from 'react';

type Theme = 'dark' | 'light';

/**
 * Dark and light, with the reader's system as the default and their choice winning over it.
 *
 * THE THREE STATES ARE DELIBERATE AND ONLY TWO ARE STORED. Nothing in `localStorage` means "use
 * whatever the system says", and the stylesheet handles that through a `prefers-color-scheme`
 * query guarded by `:not([data-theme='dark'])`. Once someone picks, `data-theme` is written to
 * `<html>` and the guard stops the system overriding them. A toggle that forgets the system
 * default is the one that gives a light-mode reader a dark flash on every visit.
 *
 * The storage read is wrapped because `localStorage` throws rather than returning null in a
 * private window with site data blocked, and a theme toggle is not worth a blank page.
 *
 * The flash is prevented by a script in the document head, not here: by the time React has
 * hydrated, the wrong theme has already been painted. See `layout.tsx`.
 */
export default function ThemeToggle() {
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => {
    // Read what the no-flash script already decided, so the button agrees with the page.
    const attr = document.documentElement.getAttribute('data-theme');
    if (attr === 'dark' || attr === 'light') {
      setTheme(attr);
      return;
    }
    setTheme(window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  }, []);

  function choose(next: Theme) {
    setTheme(next);
    document.documentElement.setAttribute('data-theme', next);
    try {
      window.localStorage.setItem('senbon-theme', next);
    } catch {
      // A blocked or full store costs this reader the preference on their next visit and
      // nothing else. The page in front of them has already changed.
    }
  }

  // Rendered as a placeholder until the effect runs, so the markup is the same on the server and
  // on the first client paint. The width is fixed so the navigation does not shift when it fills.
  const label = theme === 'light' ? 'Switch to dark' : 'Switch to light';

  return (
    <button
      type="button"
      onClick={() => choose(theme === 'light' ? 'dark' : 'light')}
      aria-label={label}
      title={label}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        width: '2rem',
        height: '2rem',
        padding: 0,
        borderRadius: '999px',
        border: '1px solid var(--rule)',
        background: 'transparent',
        color: 'var(--ink-dim)',
        cursor: 'pointer',
        flex: '0 0 auto',
      }}
    >
      {/* Both glyphs are in the markup and CSS chooses, so the icon is right on the first paint
          rather than after hydration. */}
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true">
        <g className="icon-moon" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8Z" />
        </g>
        <g className="icon-sun" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" />
        </g>
      </svg>
    </button>
  );
}
