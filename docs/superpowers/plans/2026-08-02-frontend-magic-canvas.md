# Magic Canvas Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a polished English React interface for automated/manual image extraction, individual component preview/download, and draggable canvas editing against the existing FastAPI endpoint.

**Architecture:** `App` owns the upload-to-workspace state transition. Focused upload and workspace components consume shared keyword/image helpers and a single API adapter; Fabric.js remains isolated inside `CanvasComponent`. Vitest and Testing Library cover user-observable branching and modal behavior, while Vite build, ESLint, and browser QA cover integration and layout.

**Tech Stack:** React 18, Vite 5, Tailwind CSS 4, Fabric.js 7, Vitest, Testing Library, jsdom.

## Global Constraints

- Keep `POST /api/process-image` and its current request/response contract unchanged.
- All user-facing frontend text must be English.
- Automated mode omits the `keywords` form field.
- Manual syntax supports comma-separated keywords, double-quoted multi-word keywords, and at most 10 entries.
- Every background/component preview has one action labeled exactly `Download`.
- Keep every returned layer as an independent browser item and Fabric.js object.
- Work only on `codex/frontend-magic-canvas`.

---

### Task 1: Test Harness, Keyword Parsing, and API Boundary

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/package-lock.json`
- Modify: `frontend/src/api.js`
- Create: `frontend/src/lib/keywords.js`
- Create: `frontend/src/lib/image.js`
- Create: `frontend/src/lib/keywords.test.js`
- Create: `frontend/src/api.test.js`

**Interfaces:**
- Produces: `parseKeywords(value): string[]`, `validateManualKeywords(value): { keywords: string[], error: string }`, `toImageDataUrl(value): string`, `downloadDataUrl(value, filename): void`.
- Produces: `processImage(file, keywords = null, fetchImpl = fetch): Promise<ProcessResponse>`; `keywords` is omitted when `null`.

- [ ] **Step 1: Install the test dependencies and add scripts**

Run:

```powershell
npm.cmd install --save-dev vitest @testing-library/react @testing-library/user-event @testing-library/jest-dom jsdom
```

Add `"test": "vitest run"` to `scripts`.

- [ ] **Step 2: Write failing keyword and API tests**

Cover these literal outcomes:

```js
expect(parseKeywords('dog, "wooden chair", plant')).toEqual([
  'dog', 'wooden chair', 'plant',
])
expect(validateManualKeywords('')).toEqual({
  keywords: [], error: 'Enter at least one keyword.',
})
expect(validateManualKeywords('a,b,c,d,e,f,g,h,i,j,k').error).toBe(
  'Use no more than 10 keywords.',
)
```

For `processImage`, inject a recording `fetchImpl`, inspect the real `FormData`, and verify automated submission has `file` but no `keywords`; manual submission has normalized comma-separated keywords.

- [ ] **Step 3: Run the focused tests and verify RED**

Run: `npm.cmd test -- src/lib/keywords.test.js src/api.test.js`

Expected: FAIL because helper exports and the conditional request behavior do not exist.

- [ ] **Step 4: Implement the minimal helpers and API adapter**

Implement a quote-aware comma parser, validation for empty and more than 10 values, base64-to-data-URL conversion, safe download filenames, status-to-English error mapping, and conditional `FormData` construction.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run: `npm.cmd test -- src/lib/keywords.test.js src/api.test.js`

Expected: all focused tests pass.

### Task 2: Upload and Extraction Mode Experience

**Files:**
- Create: `frontend/src/components/ExtractionModeSelector.jsx`
- Create: `frontend/src/components/UploadScreen.jsx`
- Create: `frontend/src/components/UploadScreen.test.jsx`
- Modify: `frontend/src/App.jsx`

**Interfaces:**
- Consumes: `validateManualKeywords` and `processImage` from Task 1.
- Produces: `<UploadScreen onProcessed(result) />` with automated/manual mode, file selection, loading, and English validation/errors.

- [ ] **Step 1: Write failing interaction tests**

Use Testing Library to prove:

```jsx
expect(screen.queryByLabelText('Object keywords')).not.toBeInTheDocument()
await user.click(screen.getByRole('button', { name: 'Manual Layer Extraction' }))
expect(screen.getByLabelText('Object keywords')).toBeInTheDocument()
```

Add tests that automated submission calls the injected request with `null` keywords, manual mode passes normalized keywords, and the loading state disables the action.

- [ ] **Step 2: Run the upload tests and verify RED**

Run: `npm.cmd test -- src/components/UploadScreen.test.jsx`

Expected: FAIL because the components do not exist.

- [ ] **Step 3: Implement the upload screen**

Build the accepted layout with drag/drop, file picker, selected image preview, two exclusive mode cards, conditional keyword input and adjacent instruction panel, English errors, and `Extract layers` loading behavior.

- [ ] **Step 4: Connect `App` to the upload/result transition**

Replace the monolithic upload implementation in `App.jsx` with screen state and callbacks. Do not change the backend contract.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `npm.cmd test -- src/components/UploadScreen.test.jsx`

Expected: all upload tests pass.

### Task 3: Component Browser and Enlarged Download Preview

**Files:**
- Create: `frontend/src/components/ComponentBrowser.jsx`
- Create: `frontend/src/components/ComponentPreviewModal.jsx`
- Create: `frontend/src/components/Workspace.jsx`
- Create: `frontend/src/components/Workspace.test.jsx`
- Modify: `frontend/src/App.jsx`

**Interfaces:**
- Consumes: the unchanged `ProcessResponse` shape and image/download helpers from Task 1.
- Produces: `<Workspace data onBack />`; selecting a browser row opens `ComponentPreviewModal` with the original item asset.

- [ ] **Step 1: Write failing workspace tests**

Use a complete response fixture containing background and two layers. Assert three browser rows render, clicking `Dog` opens a dialog with the dog image, and the only visible download action is named exactly `Download`.

- [ ] **Step 2: Run workspace tests and verify RED**

Run: `npm.cmd test -- src/components/Workspace.test.jsx`

Expected: FAIL because workspace components do not exist.

- [ ] **Step 3: Implement browser, modal, and workspace shell**

Create stable unique row keys using response index, render checkerboard thumbnails, open a large accessible dialog, close it through the close button or backdrop, and download the selected original PNG with a sanitized filename.

- [ ] **Step 4: Connect the workspace to `App`**

Render `Workspace` after successful processing and return to `UploadScreen` without a page reload.

- [ ] **Step 5: Run workspace and full tests and verify GREEN**

Run: `npm.cmd test`

Expected: all tests pass.

### Task 4: Responsive Fabric Canvas and Visual System

**Files:**
- Modify: `frontend/src/CanvasComponent.jsx`
- Modify: `frontend/src/index.css`
- Modify: `frontend/src/components/Workspace.jsx`
- Modify: `frontend/index.html`

**Interfaces:**
- Consumes: existing `ProcessResponse` canvas data.
- Produces: responsive Fabric editor with selection, dragging, stacking order, and original-resolution PNG export.

- [ ] **Step 1: Add failing tests for deterministic canvas sizing helpers**

Extract and test `fitCanvasSize(originalWidth, originalHeight, availableWidth, availableHeight)` with literal landscape, portrait, and no-upscale expectations.

- [ ] **Step 2: Run the helper test and verify RED**

Run: `npm.cmd test -- src/lib/canvas.test.js`

Expected: FAIL because `fitCanvasSize` does not exist.

- [ ] **Step 3: Implement responsive canvas lifecycle**

Use `ResizeObserver` on the canvas host instead of fixed `window.innerWidth` offsets. Preserve aspect ratio, clean up Fabric instances and observers, keep the background at the back, synchronize canvas selection to workspace state, and export with the inverse display scale.

- [ ] **Step 4: Implement the accepted visual design**

Define the warm neutral/green design tokens and all responsive states in `index.css`. Style the upload flow, two-column workspace, compact toolbar, checkerboard previews, and modal. Use semantic buttons and visible focus states. Replace the default Vite title/favicon identity in `index.html`.

- [ ] **Step 5: Run all automated verification**

Run:

```powershell
npm.cmd test
npm.cmd run lint
npm.cmd run build
```

Expected: tests, ESLint, and production build all exit successfully.

### Task 5: Browser QA and Final Cleanup

**Files:**
- Modify only files implicated by QA defects.

**Interfaces:**
- Consumes: the complete frontend.
- Produces: visually verified desktop/mobile upload and workspace states.

- [ ] **Step 1: Start Vite and inspect the upload screen**

Run `npm.cmd run dev -- --host 127.0.0.1` and verify desktop and narrow viewport layouts, drag target, mode switching, keyword guidance, focus visibility, and English copy.

- [ ] **Step 2: Inspect the result workspace with a controlled fixture**

Exercise background/component rows, enlarged preview, exact `Download` label, backdrop/close dismissal, draggable objects, stacking controls, and canvas export.

- [ ] **Step 3: Fix every observed defect test-first where behavior changes**

Add a failing regression test for behavior defects, confirm RED, make the minimal fix, and confirm GREEN. Pure spacing changes are verified visually.

- [ ] **Step 4: Run final verification**

Run:

```powershell
npm.cmd test
npm.cmd run lint
npm.cmd run build
git diff --check
git status --short
```

Expected: all commands pass; only intended plan and frontend changes remain.
