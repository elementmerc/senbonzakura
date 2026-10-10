// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Daniel Iwugo <ops@themalwarefiles.com>
// Author:  Daniel Iwugo
// Comment: Christ is King

import Mark from '@/components/Mark';

const REPO = 'https://github.com/elementmerc/senbonzakura';
const DOCS = 'https://elementmerc.github.io/senbonzakura';

/**
 * ALL THREE LICENCE ROWS, in the footer, because one of them surprises people.
 *
 * Two readers with no knowledge of this project went looking for the licensing position in
 * September 2026 and found it stated nowhere but the licence file itself, which is the wrong
 * place for the one fact a commercial reader most needs. Three separate things are licensed
 * differently and a reader who assumes one answer covers all three gets it wrong:
 *
 *   the code         AGPL-3.0-or-later
 *   the bundled evaluation track    CC BY-NC 4.0, so non-commercial
 *   a model you edit                whatever the base model says
 *
 * The middle one is the trap. A firm can use this tool commercially without trouble, because the
 * network clause only bites somebody reselling it as a service, but `--track default` is
 * non-commercial and they have to bring their own corpus. Neither competitor states anything
 * comparable, and one of them has no legal links at all.
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

        {/* The three licences, as a table rather than a sentence, because they genuinely differ
            and a reader skimming one line takes the first answer for all three. */}
        <div
          style={{
            border: '1px solid var(--rule)',
            borderRadius: '10px',
            overflow: 'hidden',
            maxWidth: '46rem',
            marginBottom: '2.4rem',
          }}
        >
          {[
            ['The code', 'AGPL-3.0-or-later', 'Using it inside your company triggers nothing. Reselling it as a service does.'],
            [
              'The bundled evaluation track',
              'CC BY-NC 4.0',
              'Non-commercial. For commercial work, point --track at a corpus of your own.',
            ],
            [
              'A model you edit',
              "the base model's licence",
              'Redistributing an edited checkpoint is governed by Qwen, Llama, Gemma and so on, never by us.',
            ],
          ].map(([what, licence, note], i) => (
            <div
              key={what}
              style={{
                padding: '0.95rem 1.2rem',
                borderTop: i ? '1px solid var(--rule)' : undefined,
                display: 'grid',
                gridTemplateColumns: 'minmax(11rem, 14rem) 1fr',
                gap: '0.4rem 1.2rem',
                alignItems: 'baseline',
              }}
            >
              <span style={{ fontSize: '0.9rem' }}>{what}</span>
              <span>
                <span className="mono" style={{ fontSize: '0.86rem', color: 'var(--measured)' }}>
                  {licence}
                </span>
                <span
                  style={{
                    display: 'block',
                    fontSize: 'var(--t-small)',
                    color: 'var(--ink-faint)',
                    marginTop: '0.25rem',
                    lineHeight: 1.5,
                  }}
                >
                  {note}
                </span>
              </span>
            </div>
          ))}
        </div>

        <p style={{ margin: 0, fontSize: 'var(--t-small)', color: 'var(--ink-faint)', lineHeight: 1.8 }}>
          Named for Byakuya Kuchiki's zanpakutō, the sword that scatters into a thousand blades.
          <br />
          &copy; 2026 Daniel Iwugo. This site sets no cookies, runs no analytics and loads nothing
          from anybody else.
        </p>
      </footer>
    </>
  );
}
