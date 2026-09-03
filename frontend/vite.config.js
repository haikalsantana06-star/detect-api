import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/detect': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
