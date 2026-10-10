// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

/**
 * Static export, on purpose.
 *
 * This page makes a claim to a security-conscious audience about the tool it describes: that the
 * tool makes no network call you did not ask for. A site that ships a server, sets cookies, or
 * calls an analytics endpoint on load reads as a contradiction of its own headline, whatever the
 * small print says. `output: 'export'` makes that structural rather than a promise: the build
 * produces files, there is nothing to run, and there is no request path for anything to be
 * recorded on.
 *
 * It also means the host is interchangeable, so the site cannot become a reason to keep paying
 * somebody.
 *
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  output: 'export',

  // The exporter cannot use the on-demand image optimiser, because that needs a server. Images
  // here are hand-sized instead.
  images: { unoptimized: true },

  // A trailing slash keeps the output directory layout working on any static host without
  // per-host rewrite rules, which is the thing that usually breaks a move between them.
  trailingSlash: true,

  // Fail the build on a type error rather than shipping one. The defaults already do this; it is
  // written out so that a future "just get it deployed" moment has to delete a line that says why.
  typescript: { ignoreBuildErrors: false },
};

export default nextConfig;
