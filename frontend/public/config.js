// Runtime API configuration.
//
// In a container this file is overwritten at start-up from $API_BASE_URL (see
// docker-entrypoint.d/10-furina-config.sh). During `npm run dev` it stays as
// shipped, and services.ts falls through to VITE_API_BASE_URL, then to the
// same-origin default of "/api" -- which the Vite dev proxy forwards to Django.
//
// window.__FURINA_API_BASE_URL__ = "http://localhost:8000/api";
