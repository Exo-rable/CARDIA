# Cardia frontend

```bash
npm ci
npm run dev
```

Vite runs at http://localhost:5173. During development, the API defaults to http://localhost:8000. Set `VITE_API_URL` in a frontend `.env.local` file if the API runs elsewhere.

For production, `npm run build` creates the static site in `dist/`. Leave `VITE_API_URL` empty when the web server proxies `/api/` to FastAPI on the same origin; set it to the API's HTTPS origin when the services are hosted separately. The value is embedded at build time.
