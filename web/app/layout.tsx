// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

import type { Metadata } from 'next';
import type { ReactNode } from 'react';
import './globals.css';

export const metadata: Metadata = {
  title: 'Senbonzakura',
  description:
    'Measure what a behaviour edit did to an open-weight model, and what it cost. Runs on your machine, makes no network call you did not ask for.',
  metadataBase: new URL('https://senbon.dev'),
  openGraph: {
    title: 'Senbonzakura',
    description:
      'Measure what a behaviour edit did to an open-weight model, and what it cost.',
    url: 'https://senbon.dev',
    siteName: 'Senbonzakura',
    type: 'website',
  },
  // No analytics, no verification tokens, no third-party script. The page claims the tool makes
  // no call you did not ask for, and a tracker here would be the page contradicting its headline.
  robots: { index: true, follow: true },
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en-GB">
      <body>{children}</body>
    </html>
  );
}
