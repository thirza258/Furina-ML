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
      },
    },
  },
  plugins: [],
}

