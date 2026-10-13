// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

/**
 * WHICH RELEASE THIS SITE DESCRIBES, in one place.
 *
 * WHY IT IS A CONSTANT AND NOT SIX STRING LITERALS. It was six string literals, across six
 * components, and 0.5.0 shipped on 2026-10-13 without any of them changing. So the live site went
 * on advertising 0.4.1 on PyPI, linking the v0.4.1 Colab notebook, and pointing its licence,
 * acceptable-use, contributing and changelog links at a tag two releases old.
 *
 * That is not only stale. 0.4.0 and 0.4.1 are being yanked from PyPI for a token written into a
 * generated model card, so every one of those links sent a reader to a release we are in the
 * middle of withdrawing, from the page whose whole argument is that this project's numbers can be
 * trusted.
 *
 * A version in six places is a release step somebody has to remember. A version in one place with
 * a test holding it to the project's own released version is a release step that cannot be
 * forgotten, which is the difference this project keeps paying for.
 */
export const RELEASE = '0.5.1';

/** The tag that names it on GitHub. */
export const TAG = `v${RELEASE}`;

export const REPO = 'https://github.com/elementmerc/senbonzakura';

/** A file in the repository, pinned to this release rather than to a moving branch. */
export function atTag(path: string): string {
  return `${REPO}/blob/${TAG}/${path}`;
}
