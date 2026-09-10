// Senbonzakura documentation site.
//
// Built under hephaestus ADR 75: every project with readers who are not its author ships built
// documentation. senbonzakura is public and AGPL, so this deploys to GitHub Pages. npm lives in
// this directory and nowhere else; nothing here reaches the Python wheel, which is built from
// `src/` alone.
//
// `ignoreDeadLinks` is deliberately NOT set. The ADR requires the docs build to fail on a broken
// link, and a build that warns instead of failing is a build nobody reads the output of.

const BASE = process.env.DOCS_BASE || '/senbonzakura/'
// og:image has to be an absolute URL: a social card served from a relative path is fetched by a
// crawler that has no page context and simply does not resolve, so the preview falls back to
// nothing and the failure is invisible from inside the site.
const SITE = process.env.DOCS_SITE || 'https://elementmerc.github.io'

export default {
  title: 'Senbonzakura',
  description:
    'Precision abliteration, with receipts. Multi-direction refusal abliteration for transformer '
    + 'language models, and the instruments to tell whether it worked.',
  lang: 'en-GB',
  base: BASE,
  cleanUrls: true,
  lastUpdated: true,

  // Markdown that lives under docs/ without being part of the site.
  //
  // `writeups/` are articles with their own covers and their own publication route; they are not
  // reference material and they read in a different register. `private/` never appears here at
  // all (it is excluded from the repository), but the pattern stays as a belt-and-braces guard
  // against anything internal being dropped into this tree and silently published: this site is
  // public, and a build that quietly renders whatever it finds is one careless `mv` from
  // publishing a working note.
  srcExclude: ['writeups/**', 'private/**', '**/VOICE.md', 'node_modules/**'],

  head: [
    // SVG first for anything modern, .ico for the browsers and feed readers that still ask for
    // one by that name, and a 32px PNG for the ones that do neither.
    ['link', { rel: 'icon', href: `${BASE}favicon.svg`, type: 'image/svg+xml' }],
    ['link', { rel: 'icon', href: `${BASE}favicon.ico`, sizes: '48x48' }],
    ['link', { rel: 'icon', href: `${BASE}favicon-32.png`, type: 'image/png', sizes: '32x32' }],
    ['link', { rel: 'apple-touch-icon', href: `${BASE}apple-touch-icon.png`, sizes: '180x180' }],
    ['link', { rel: 'manifest', href: `${BASE}manifest.webmanifest` }],
    // Deep navy, the brand's dark ground. It tints the browser chrome on mobile, so a value that
    // does not match the page reads as a rendering fault rather than a choice.
    ['meta', { name: 'theme-color', content: '#0E1330' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'Senbonzakura' }],
    ['meta', {
      property: 'og:description',
      content: 'Precision abliteration, with receipts. Multi-direction refusal abliteration, and '
        + 'the instruments to tell whether it worked.',
    }],
    ['meta', { property: 'og:image', content: `${SITE}${BASE}social-card.png` }],
    ['meta', { name: 'twitter:card', content: 'summary_large_image' }],
    ['meta', { name: 'twitter:image', content: `${SITE}${BASE}social-card.png` }],
  ],

  themeConfig: {
    logo: '/mark.svg',
    siteTitle: 'Senbonzakura',

    nav: [
      { text: 'Guide', link: '/guide/what-it-is', activeMatch: '/guide/' },
      { text: 'Reference', link: '/reference/cli', activeMatch: '/reference/' },
      { text: 'Contribute', link: '/contributing' },
      {
        text: 'v0.4 (unreleased)',
        items: [
          { text: 'Changelog', link: 'https://github.com/elementmerc/senbonzakura/blob/main/CHANGELOG.md' },
          { text: 'Licence (AGPL-3.0-or-later)', link: 'https://github.com/elementmerc/senbonzakura/blob/main/LICENSE' },
        ],
      },
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Start here',
          items: [
            { text: 'Quickstart', link: '/guide/quickstart' },
            { text: 'What it is', link: '/guide/what-it-is' },
            { text: 'Install', link: '/guide/install' },
            { text: 'Your first run', link: '/guide/first-run' },
          ],
        },
        {
          text: 'How it works',
          items: [
            { text: 'The method', link: '/guide/how-it-works' },
            { text: 'The track', link: '/guide/the-track' },
            { text: 'Contamination', link: '/guide/contamination' },
            { text: 'Quantisation', link: '/guide/quantisation' },
            { text: 'The evaluation track: dataset card', link: '/evaluation-track-card' },
          ],
        },
        {
          text: 'Measuring',
          items: [
            { text: 'The compass', link: '/guide/compass' },
            { text: 'Benchmarking against another tool', link: '/guide/benchmark' },
            { text: 'How this compares to other tools', link: '/comparison' },
          ],
        },
        {
          text: 'Read this before quoting a number',
          items: [
            { text: 'What is and is not established', link: '/guide/what-we-know' },
            { text: 'Who got here first', link: '/guide/prior-art' },
            { text: 'Limits and known defects', link: '/guide/limits' },
          ],
        },
      ],
      '/reference/': [
        {
          text: 'Reference',
          items: [
            { text: 'Commands', link: '/reference/cli' },
            { text: 'Flags worth knowing', link: '/reference/flags' },
            { text: 'Which models it can open', link: '/reference/models' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/elementmerc/senbonzakura' },
    ],

    editLink: {
      pattern: 'https://github.com/elementmerc/senbonzakura/edit/main/docs/:path',
      text: 'Suggest a change to this page',
    },

    outline: { level: [2, 3], label: 'On this page' },

    footer: {
      message:
        'AGPL-3.0-or-later. A modified work based in part on '
        + '<a href="https://github.com/p-e-w/heretic">Heretic</a>.',
      copyright: '© 2026 Daniel Iwugo',
    },

    search: { provider: 'local' },

    docFooter: { prev: 'Previous', next: 'Next' },
  },

  markdown: {
    lineNumbers: false,
    theme: { light: 'github-light', dark: 'github-dark' },
  },
}
