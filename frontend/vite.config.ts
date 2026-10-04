import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  cacheDir: process.env.VITE_CACHE_DIR || 'node_modules/.vite',
  server: { proxy: Object.fromEntries(['/api', '/docs', '/openapi.json'].map(path => [path, process.env.API_PROXY_TARGET || 'http://127.0.0.1:8000'])) },
  build: { rollupOptions: { output: { manualChunks: { charts: ['recharts'], vendor: ['react', 'react-dom', 'react-router-dom'] } } } },
})
