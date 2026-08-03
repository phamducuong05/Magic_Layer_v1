# LAN Server Access Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the frontend and backend reachable from LAN devices through `192.168.1.20`.

**Architecture:** Preserve the current two-port frontend/backend architecture. Bind Vite to all interfaces, point its default API URL at the server LAN address, and explicitly authorize the LAN frontend origin in FastAPI CORS.

**Tech Stack:** Vite, React, Vitest, FastAPI, pytest

## Global Constraints

- LAN frontend URL is `http://192.168.1.20:5173`.
- LAN backend URL is `http://192.168.1.20:8009`.
- Existing localhost support and `VITE_API_URL` override remain available.
- Image-processing and API behavior remain unchanged.

---

### Task 1: Configure and verify LAN access

**Files:**
- Modify: `frontend/vite.config.js`
- Modify: `frontend/src/api.js`
- Modify: `backend/main.py`
- Test: `frontend/src/lan-config.test.js`
- Test: `tests/test_lan_server_config.py`

**Interfaces:**
- Consumes: Vite server configuration, `VITE_API_URL`, FastAPI middleware.
- Produces: Vite listener `0.0.0.0:5173`, fallback API origin `http://192.168.1.20:8009`, and LAN CORS authorization.

- [ ] Write tests asserting the listener, fallback URL, and CORS origin.
- [ ] Run the focused tests and confirm they fail because LAN configuration is absent.
- [ ] Add the three minimal configuration changes.
- [ ] Run focused tests, frontend tests, and relevant backend tests.
- [ ] Run `git diff --check` and confirm processing code has no diff.
