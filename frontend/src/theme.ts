import { createTheme } from '@mui/material/styles';
// Type-only side-effect import: it teaches MUI's `components` map about
// `MuiDataGrid`, which lives in the X package and is otherwise unknown to it.
import type {} from '@mui/x-data-grid/themeAugmentation';

/**
 * MUI's slice of the academic theme.
 *
 * Only the DataGrid (large table previews) and a few CssBaseline defaults come
 * from MUI, but they have to sit on the same paper as everything Tailwind
 * draws. The values here are the Tailwind tokens by hand -- MUI cannot read
 * `tailwind.config.js`, so any change there has to be mirrored here.
 */
const ink = '#1A1815';
const inkMuted = '#5A544B';
const rule = '#D8D3C7';
const paper = '#FAF9F6';
const paperRaised = '#FFFFFF';
const paperSunken = '#F2F0EA';

const sans =
  '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
const serif =
  'Georgia, "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua", "Times New Roman", serif';

const theme = createTheme({
  palette: {
    primary: { main: '#5A1D2E' },   // Tailwind 'primary'  (burgundy)
    secondary: { main: '#1F3352' },  // Tailwind 'secondary' (navy)
    info: { main: '#8C6A1F' },       // Tailwind 'accent'    (brass)
    error: { main: '#B3261E' },      // Tailwind 'state-bad'
    success: { main: '#1F7A5A' },    // Tailwind 'state-good'
    warning: { main: '#8A5A0B' },    // Tailwind 'state-warn'
    background: { default: paper, paper: paperRaised },
    text: { primary: ink, secondary: inkMuted },
    divider: rule,
  },
  typography: {
    fontFamily: sans,
    // Headings are the one place MUI should speak in the serif.
    h1: { fontFamily: serif },
    h2: { fontFamily: serif },
    h3: { fontFamily: serif },
    h4: { fontFamily: serif },
    h5: { fontFamily: serif },
    h6: { fontFamily: serif },
    button: { textTransform: 'none', fontWeight: 500 },
  },
  shape: {
    // Square, like the rest of the page. Controls keep 2px in Tailwind.
    borderRadius: 2,
  },
  components: {
    MuiPaper: {
      // Drop shadows are the thing this theme is defined against; a hairline
      // does the same separating work without the floating-card effect.
      defaultProps: { elevation: 0 },
      styleOverrides: {
        root: { backgroundImage: 'none', border: `1px solid ${rule}` },
      },
    },
    MuiDataGrid: {
      styleOverrides: {
        root: {
          border: `1px solid ${rule}`,
          borderRadius: 0,
          fontFamily: sans,
          fontSize: '0.8125rem',
          color: ink,
          fontVariantNumeric: 'tabular-nums',
        },
        columnHeaders: {
          backgroundColor: paperSunken,
          borderBottom: `1px solid ${rule}`,
        },
        columnHeaderTitle: {
          fontWeight: 600,
          fontSize: '0.75rem',
          letterSpacing: '0.06em',
          textTransform: 'uppercase',
          color: inkMuted,
        },
        cell: { borderBottom: `1px solid ${rule}` },
        footerContainer: { borderTop: `1px solid ${rule}` },
      },
    },
  },
});

export default theme;
