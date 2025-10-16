/** @type {import('tailwindcss').Config} */
const config = {
  darkMode: 'class', // allows <html class="dark">
  theme: {
    extend: {
      colors: {
        purple: {
          600: '#7c3aed',
          700: '#6d28d9',
        },
      },
    },
  },
  content: [
    './app/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
  ],
};

export default config;
