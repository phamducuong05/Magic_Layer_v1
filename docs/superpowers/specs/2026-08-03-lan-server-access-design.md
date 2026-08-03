# LAN Server Access Design

## Goal

Allow devices on the same LAN to open the application at
`http://192.168.1.20:5173` and reach the backend at
`http://192.168.1.20:8009`.

## Changes

- Configure Vite to listen on `0.0.0.0:5173`.
- Keep `VITE_API_URL` authoritative and change only its fallback to
  `http://192.168.1.20:8009`.
- Add `http://192.168.1.20:5173` to FastAPI CORS origins while preserving both
  localhost origins.
- Do not change API routes, image processing, model loading, layer extraction,
  job handling, or response schemas.

## Runtime Requirement

The backend must be started with `--host 0.0.0.0 --port 8009`. Windows Firewall
must allow inbound TCP traffic on ports 5173 and 8009.

## Verification

Automated tests will assert the Vite listener, default API origin, and complete
CORS origin set. Existing frontend and focused backend tests will be rerun.
