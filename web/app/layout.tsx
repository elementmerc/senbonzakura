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
 * that their weights never leave their machine. A page that quietly calls a third party while
 * making a claim about where data goes reads as the claim being decorative.
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
    'Measure what a behaviour edit did to an open-weight model, and what it cost. The measuring runs on your own hardware and your weights never leave it.',
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
  // The browser chrome and the page are one surface rather than a page in a frame of the wrong
  // colour, and there are TWO entries because the chrome has to follow the theme as well. A
  // single themeColor was the brand navy, which left a navy address bar above a neutral page,
  // the same mistake the navigation bar was making.
  //
  // These track the system rather than a stored choice, which is all the media query can see. A
  // reader who overrides the system on this site gets the right page and a chrome one shade out;
  // there is no metadata form that can read localStorage, and the alternative is a chrome that is
  // wrong for everybody who has not chosen.
  themeColor: [
    { media: '(prefers-color-scheme: dark)', color: '#0E1014' },
    { media: '(prefers-color-scheme: light)', color: '#F2F1EF' },
  ],
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en-GB" className={`${body.variable} ${mono.variable}`} suppressHydrationWarning>
      <head>
        {/* THE THEME IS SET BEFORE THE FIRST PAINT, and it has to be done here rather than in a
            component. React has not hydrated when the browser paints the first frame, so a theme
            applied in an effect arrives after the reader has already seen the other one. That
            flash is the single most noticeable defect a theme toggle has.

            Only a stored CHOICE is applied. Absence means "follow the system", which the
            stylesheet handles with a `prefers-color-scheme` query, so this script writes nothing
            when nobody has chosen and the markup stays identical for every first-time visitor.

            Wrapped in try/catch because `localStorage` throws rather than returning null in a
            private window with site data blocked, and an exception here runs before anything is
            on screen, which would leave the page blank. */}
        <script
          dangerouslySetInnerHTML={{
            __html:
              "try{var t=localStorage.getItem('senbon-theme');" +
              "if(t==='dark'||t==='light')document.documentElement.setAttribute('data-theme',t);}catch(e){}",
          }}
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
