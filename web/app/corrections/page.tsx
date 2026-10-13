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
        {/* The breadcrumb and the standfirst that stood here were removed by the operator on
            2026-10-13. The page now opens on its own heading, which says the same thing in fewer
            words: "Twelve things we published that were not true." */}
        <Corrections />
      </main>
      <Footer />
    </>
  );
}
