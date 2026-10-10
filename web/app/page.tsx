// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

import Footer from '@/components/Footer';
import Hero from '@/components/Hero';
import Honesty from '@/components/Honesty';
import Install from '@/components/Install';
import Measured from '@/components/Measured';
import Nav from '@/components/Nav';
import TheGap from '@/components/TheGap';

/**
 * Six blocks, and the order is the argument:
 *
 *   1. Hero       what this is, and the promise no hosted competitor can make
 *   2. The gap    two other groups' measurements of the problem, none of them ours
 *   3. Measured   our own figures, and the headline one is our flagship feature losing
 *   4. Honesty    the corrections, kept published, which nobody else here does
 *   5. Install    the one command, no timing claim, because nobody has measured one
 *   6. Footer     all three licence rows, because they genuinely differ
 *
 * Deliberately absent: a pricing page, because the commercial pack is designed and not built,
 * and naming an unreleased thing publicly is a promise about a timeline we may want to change.
 * And no exploit-benchmark strip, which is the obvious thing to copy from the nearest competitor
 * and the one framing that turns a measurement tool into something else.
 */
export default function Page() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <TheGap />
        <Measured />
        <Honesty />
        <div id="install">
          <Install />
        </div>
      </main>
      <Footer />
    </>
  );
}
