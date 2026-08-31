/** @type {import('tailwindcss').Config} */

// The type stack is deliberately system-only: this app is deployed from a
// self-contained Docker image that may have no outbound internet, so a webfont
// request would simply hang. Georgia and its Palatino-family fallbacks are the
// old-style serifs that ship everywhere, and they are what gives the app its
// typeset-on-paper voice.
const serif = [
  'Georgia',
  'Iowan Old Style',
  'Palatino Linotype',
  'Palatino',
  'Book Antiqua',
  'Times New Roman',
  'serif',
];

const sans = [
  '-apple-system',
  'BlinkMacSystemFont',
  'Segoe UI',
  'Roboto',
  'Helvetica Neue',
  'Arial',
  'sans-serif',
];

const mono = [
  'ui-monospace',
  'SFMono-Regular',
  'SF Mono',
  'Menlo',
  'Consolas',
  'Liberation Mono',
  'monospace',
];

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        // Headings and running prose. The reading voice of the app.
        serif,
        // Controls, labels, table headers -- anything you operate rather than read.
        sans,
        // Column names, metric values, hyperparameters: things that must line up.
        mono,
      },
      colors: {
        // --- brand ------------------------------------------------------
        // Deep burgundy and Oxford navy: the two colours academic presses
        // have used on spines and title pages for a century. Both clear 10:1
        // against paper, so they can carry body-weight text as well as fills.
        'primary': '#5A1D2E',
        'secondary': '#1F3352',
        // Brass. Used sparingly, and never as the only signal.
        'accent': '#8C6A1F',

        // --- surfaces ---------------------------------------------------
        // Warm off-white, the colour of uncoated book stock. A pure #FFF page
        // with hairline rules reads as a web form; this reads as a page.
        'paper': '#FAF9F6',
        'paper-raised': '#FFFFFF',
        'paper-sunken': '#F2F0EA',
        // Hairlines. Warm greys, so they sit with the paper rather than
        // cutting across it the way a blue-grey border does.
        'rule': '#D8D3C7',
        'rule-strong': '#8C857A',

        // Chart tokens. These are NOT the brand colors: the brand pair fails
        // the categorical lightness/chroma checks, so these are the nearest
        // steps that pass (CVD deltaE 17.5 deutan / 19.8 normal vision).
        // Do not substitute `primary`/`secondary` for series color.
        'series-1': '#5B63E0',   // first categorical slot (e.g. "Train")
        'series-2': '#38A9C8',   // second categorical slot (e.g. "Test")

        // Single-hue sequential ramp, light -> dark, lightness-monotonic.
        // Steps 0-4 take ink text; steps 5-6 take white text.
        'ramp-0': '#EAF6FA',
        'ramp-1': '#C5E6F0',
        'ramp-2': '#93CFE3',
        'ramp-3': '#5FB4D0',
        'ramp-4': '#348FB0',
        'ramp-5': '#1F6C8B',
        'ramp-6': '#134F68',

        // Reserved status colors. Always shipped with an icon or a label,
        // never as the only signal. Never reused as a series color.
        'state-good': '#1F7A5A',
        'state-warn': '#8A5A0B',
        'state-bad': '#B3261E',

        // Warm near-black, not slate. Slate carries a blue cast that fights
        // the paper; ink sits on it the way printing ink sits on stock.
        'ink': '#1A1815',
        'ink-muted': '#5A544B',
      },
      borderRadius: {
        // Academic pages are set on a rectangular grid. Form controls keep a
        // 2px softening so they still read as operable; nothing else rounds.
        'control': '2px',
      },
      maxWidth: {
        // A comfortable measure for running prose, per the 65-75 character rule.
        'measure': '68ch',
      },
      fontSize: {
        // A conventional typographic scale rather than Tailwind's default
        // steps, so headings and captions relate to each other by ratio.
        'caption': ['0.75rem', { lineHeight: '1.4' }],
        'small': ['0.8125rem', { lineHeight: '1.5' }],
        'body': ['1rem', { lineHeight: '1.65' }],
        'lead': ['1.125rem', { lineHeight: '1.65' }],
        'h3': ['1.3125rem', { lineHeight: '1.35' }],
        'h2': ['1.75rem', { lineHeight: '1.25' }],
        'h1': ['2.375rem', { lineHeight: '1.15' }],
        'display': ['3rem', { lineHeight: '1.1' }],
      },
    },
  },
  plugins: [],
}
