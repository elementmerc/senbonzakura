// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

import type { Metadata, Viewport } from 'next';
import { Hanken_Grotesk, IBM_Plex_Mono } from 'next/font/google';
import type { ReactNode } from 'react';
import './globals.css';

/**
 * The brand kit's two faces, loaded through next/font rather than a stylesheet link.
 *
 * next/font fetches these at BUILD time and emits them as files served from this origin. That is
 * the difference that matters here rather than the speed: a Google Fonts <link> makes every
 * visitor's browser call a third party on load, and this page tells a security-conscious reader
 * that the tool makes no network call they did not ask for. A page that quietly calls out while
 * saying so reads as the claim being decorative.
 */
const body = Hanken_Grotesk({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-body',
});

const mono = IBM_Plex_Mono({
  subsets: ['latin'],
  weight: ['400', '500'],
  display: 'swap',
  variable: '--font-mono',
});

export const metadata: Metadata = {
  title: 'Senbonzakura',
  description:
    'Measure what a behaviour edit did to an open-weight model, and what it cost. Runs on your machine, makes no network call you did not ask for.',
  metadataBase: new URL('https://senbon.dev'),
  manifest: '/manifest.webmanifest',
  icons: {
    icon: [
      { url: '/favicon.svg', type: 'image/svg+xml' },
      { url: '/favicon-32.png', sizes: '32x32', type: 'image/png' },
    ],
    shortcut: '/favicon.ico',
    apple: '/apple-touch-icon.png',
  },
  openGraph: {
    title: 'Senbonzakura',
    description:
      'Measure what a behaviour edit did to an open-weight model, and what it cost.',
    url: 'https://senbon.dev',
    siteName: 'Senbonzakura',
    type: 'website',
    images: [{ url: '/social-card.png', width: 1280, height: 640 }],
  },
  twitter: {
    card: 'summary_large_image',
    images: ['/social-card.png'],
  },
  // No analytics, no verification tokens, no third-party script, for the reason given above.
  robots: { index: true, follow: true },
};

export const viewport: Viewport = {
  // Deep navy, so the browser chrome and the page are one surface rather than a dark page in a
  // light frame. It is the brand's own background colour.
  themeColor: '#0E1330',
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en-GB" className={`${body.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
