import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#75BFDB', // Tailwind 'primary'
    },
    secondary: {
      main: '#2E39A0', // Tailwind 'secondary'
    },
    info: {
      main: '#27E1E6', // Tailwind 'accent'
    },
  },
});

export default theme;
