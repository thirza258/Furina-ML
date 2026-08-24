/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'primary': '#75BFDB',
        'secondary': '#2E39A0',
        'accent': '#27E1E6',

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

        'ink': '#0F172A',
        'ink-muted': '#5B6472',
      },
    },
  },
  plugins: [],
}
