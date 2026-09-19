/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0a0e17',
        card: '#111827',
        cardHover: '#1f2937',
        primary: '#00e676',
        secondary: '#00bfa5',
        danger: '#ff5252',
        warning: '#ffc107',
        textMain: '#e8edf5',
        textMuted: '#8b97b0',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
        display: ['Space Grotesk', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
