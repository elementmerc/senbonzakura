// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

import Mark from '@/components/Mark';

const REPO = 'https://github.com/elementmerc/senbonzakura';
const DOCS = 'https://elementmerc.github.io/senbonzakura';

/**
 * Links, a copyright line, and nothing else.
 *
 * THE LICENCE TABLE THAT WAS HERE IS GONE, by the operator's decision of 2026-10-10. It set out
 * all three licences, because three separate things are licensed differently and a reader who
 * assumes one answer covers all three gets it wrong. That remains true, so the Project column
 * now carries a licence link rather than losing the question entirely: the detail belongs in the
 * repository, where it is authoritative, instead of being restated in a footer where it can
 * drift out of agreement with the files it describes.
 *
 * The no-tracking sentence is gone too, and for a better reason than brevity. It was accurate on
 * the day it was written and the project has since decided to collect telemetry, so it was a
 * claim with an expiry date sitting in the one place nobody re-reads. A promise about data
 * handling belongs in a privacy policy that is versioned and dated, not in a footer.
 */
export default function Footer() {
  return (
    <>
      <hr className="rule" />
      <footer className="band" style={{ paddingTop: 'calc(var(--band) * 0.7)', paddingBottom: 'calc(var(--band) * 0.7)' }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.6rem',
            marginBottom: '2.4rem',
          }}
        >
          <Mark size={28} />
          <span className="wordmark" style={{ fontSize: '0.86rem' }}>
            Senbonzakura
          </span>
          <span className="jp" aria-hidden="true" style={{ color: 'var(--ink-faint)' }}>
            千本桜
          </span>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(13rem, 1fr))',
            gap: '2.4rem 2rem',
            marginBottom: '3rem',
          }}
        >
          <div>
            <p className="mono" style={{ fontSize: '0.72rem', letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--ink-faint)', margin: '0 0 0.9rem' }}>
              Start
            </p>
            {[
              [`${DOCS}/guide/install`, 'Install'],
              [`${DOCS}/guide/quickstart`, 'Quickstart'],
              [`${DOCS}/guide/first-run`, 'Your first run'],
              ['https://pypi.org/project/senbonzakura/', 'PyPI'],
            ].map(([href, label]) => (
              <p key={label} style={{ margin: '0 0 0.5rem', fontSize: '0.92rem' }}>
                <a href={href} style={{ textDecoration: 'none', color: 'var(--ink-dim)' }}>
                  {label}
                </a>
              </p>
            ))}
          </div>

          <div>
            <p className="mono" style={{ fontSize: '0.72rem', letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--ink-faint)', margin: '0 0 0.9rem' }}>
              Evidence
            </p>
            {[
              [`${DOCS}/guide/what-we-know`, 'What is established'],
              [`${DOCS}/guide/what-we-got-wrong`, 'What we got wrong'],
              [`${DOCS}/guide/limits`, 'Limits and defects'],
              [`${DOCS}/guide/prior-art`, 'Prior art'],
            ].map(([href, label]) => (
              <p key={label} style={{ margin: '0 0 0.5rem', fontSize: '0.92rem' }}>
                <a href={href} style={{ textDecoration: 'none', color: 'var(--ink-dim)' }}>
                  {label}
                </a>
              </p>
            ))}
          </div>

          <div>
            <p className="mono" style={{ fontSize: '0.72rem', letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--ink-faint)', margin: '0 0 0.9rem' }}>
              Project
            </p>
            {[
              [REPO, 'Repository'],
              [`${REPO}/blob/v0.4.1/LICENSE`, 'Licence'],
              [`${REPO}/blob/v0.4.1/ACCEPTABLE-USE.md`, 'Acceptable use'],
              [`${REPO}/blob/v0.4.1/CONTRIBUTING.md`, 'Contributing'],
              [`${REPO}/blob/v0.4.1/CHANGELOG.md`, 'Changelog'],
            ].map(([href, label]) => (
              <p key={label} style={{ margin: '0 0 0.5rem', fontSize: '0.92rem' }}>
                <a href={href} style={{ textDecoration: 'none', color: 'var(--ink-dim)' }}>
                  {label}
                </a>
              </p>
            ))}
          </div>
        </div>

        <p style={{ margin: 0, fontSize: 'var(--t-small)', color: 'var(--ink-faint)', lineHeight: 1.8 }}>
          &copy; 2026 Daniel Iwugo
        </p>
      </footer>
    </>
  );
}
