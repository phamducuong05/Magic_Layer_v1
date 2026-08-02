# Frontend Visual Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a richer creative-studio interface and a responsive preview that always shows the complete isolated component.

**Architecture:** Preserve the current React component and API boundaries. Add semantic hooks to the preview component, then replace the shared CSS visual system so upload, asset browser, canvas, and modal remain responsive without changing data flow.

**Tech Stack:** React 18, Vite 5, Fabric 7, Vitest, Testing Library, CSS

## Global Constraints

- Keep all backend contracts unchanged.
- Keep all visible product copy in English.
- Keep the Download button label exactly `Download`.
- Do not add a new UI dependency.

---

### Task 1: Responsive component preview

**Files:**
- Modify: `frontend/src/components/Workspace.test.jsx`
- Modify: `frontend/src/components/ComponentPreviewModal.jsx`
- Modify: `frontend/src/index.css`

**Interfaces:**
- Consumes: workspace item `{ image, name, width, height, kind }`
- Produces: preview image with intrinsic dimensions and `.preview-object`

- [ ] Add a failing test asserting the preview image receives its source dimensions and contain-fit class.
- [ ] Run `npm test -- src/components/Workspace.test.jsx` and confirm the assertion fails.
- [ ] Add intrinsic dimensions and the preview object hook to the modal.
- [ ] Replace fixed preview height with intrinsic responsive sizing.
- [ ] Re-run the focused test and confirm it passes.

### Task 2: Creative-studio visual refresh

**Files:**
- Modify: `frontend/src/components/UploadScreen.jsx`
- Modify: `frontend/src/components/ExtractionModeSelector.jsx`
- Modify: `frontend/src/components/ComponentBrowser.jsx`
- Modify: `frontend/src/components/Workspace.jsx`
- Modify: `frontend/src/CanvasComponent.jsx`
- Modify: `frontend/src/index.css`

**Interfaces:**
- Consumes: existing component props and extraction response
- Produces: unchanged interactions with new semantic visual wrappers

- [ ] Add only the semantic wrappers and decorative elements required by the new composition.
- [ ] Replace palette, typography, surfaces, spacing, gallery, toolbar, canvas, and responsive rules.
- [ ] Preserve existing accessible names and action labels.
- [ ] Run the full test suite and fix regressions.

### Task 3: Verification and delivery

**Files:**
- Verify: `frontend/src/**`

**Interfaces:**
- Consumes: complete frontend
- Produces: verified branch update

- [ ] Run `npm test`, `npm run lint`, and `npm run build`.
- [ ] Inspect upload, manual input, workspace, and preview at desktop and mobile widths.
- [ ] Run `git diff --check` and review the final diff.
- [ ] Commit and push `codex/frontend-magic-canvas`.
