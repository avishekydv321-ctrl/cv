/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#e6fff7',
          100: '#b3ffe6',
          200: '#80ffd9',
          300: '#4dffcc',
          400: '#1affbf',
          500: '#00e6ac',
          600: '#00b386',
          700: '#008066',
          800: '#004d40',
          900: '#001a14',
        },
        dark: {
          50: '#e6edf3',
          100: '#c9d1d9',
          200: '#9fb0c3',
          300: '#7ee7c7',
          400: '#3a94ff',
          500: '#232b36',
          600: '#1b2430',
          700: '#131922',
          800: '#0e1117',
          900: '#0b1017',
          950: '#05080c',
        }
      },
      fontFamily: {
        sans: ['Inter', 'Segoe UI', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
    },
  },
  plugins: [],
};
