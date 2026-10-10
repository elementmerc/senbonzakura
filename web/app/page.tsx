// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

import Hero from '@/components/Hero';
import TheGap from '@/components/TheGap';

/**
 * Six blocks, in this order, and the order is the argument:
 *
 *   1. Hero          what this is, and the promise no hosted competitor can make
 *   2. The gap       other people's measurements of the problem                     <- built
 *   3. Measured      our own figures, each linking to the evidence file             <- next
 *   4. What we got wrong   the two honesty pages, which nobody else in this field has
 *   5. Install      the one command, and the notebook
 *   6. Footer       all three licence rows, acceptable use, repository, contact
 *
 * Deliberately absent: a pricing page, because the commercial pack is designed and not built, and
 * naming an unreleased thing publicly is a promise about a timeline we may want to change. And no
 * exploit-benchmark strip, which is the obvious thing to copy from the nearest competitor and the
 * one framing that turns a measurement tool into something else.
 */
export default function Page() {
  return (
    <main>
      <Hero />
      <TheGap />
    </main>
  );
}
