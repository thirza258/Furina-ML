import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The app asks for `/api` on its own origin (see services.ts). In the container
// nginx proxies that to Django; during `npm run dev` this proxy does the same
// job, so there is one code path instead of a dev-only absolute URL.
//
// Point it somewhere else with VITE_DEV_API_PROXY, or bypass it entirely by
// setting VITE_API_BASE_URL to an absolute URL in `.env`.
const devApiTarget = process.env.VITE_DEV_API_PROXY ?? 'http://localhost:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: devApiTarget, changeOrigin: true },
      // The admin and its static assets, so `/admin/` works on :5173 too.
      '/admin': { target: devApiTarget, changeOrigin: true },
      '/static': { target: devApiTarget, changeOrigin: true },
    },
  },
})
