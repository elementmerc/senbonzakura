// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

import type { Metadata } from 'next';
import Corrections from '@/components/Corrections';
import Footer from '@/components/Footer';
import Nav from '@/components/Nav';

/**
 * A page of its own, because this is the thing worth linking to.
 *
 * It was an anchor on the landing page, which meant the one asset neither competitor has could
 * only be cited as "senbon.dev, scroll down a bit". A correction log is the kind of thing someone
 * links to in an argument, and a link into the middle of a marketing page is a weaker object than
 * a page with its own title, its own description and its own place in a search index.
 *
 * The same component renders in both places. The landing page keeps it as a section, because a
 * visitor who never clicks anything should still meet it.
 */
export const metadata: Metadata = {
  title: 'What we got wrong — Senbonzakura',
  description:
    'Twelve things we published that were not true, kept on the site with the original wording beside the correction.',
  alternates: { canonical: '/corrections' },
};

export default function CorrectionsPage() {
  return (
    <>
      <Nav />
      <main>
        <header className="band" style={{ paddingBottom: 0 }}>
          <p
            className="mono"
            style={{
              color: 'var(--ink-faint)',
              fontSize: 'var(--t-small)',
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              margin: '0 0 1rem',
            }}
          >
            <a href="/" style={{ textDecoration: 'none' }}>
              Senbonzakura
            </a>{' '}
            / corrections
          </p>
          <p style={{ color: 'var(--ink-dim)', maxWidth: 'var(--measure)', margin: 0 }}>
            This page exists because a measurement tool that will not publish its own mistakes is
            asking to be taken on trust, which is the thing it sells against.
          </p>
        </header>
        <Corrections />
      </main>
      <Footer />
    </>
  );
}
