# Frontend Visual Refresh Design

## Goal

Refresh Magic Layers into a visually distinctive creative workspace and ensure every isolated component is fully visible in its preview at any aspect ratio.

## Visual direction

- Use a graphite studio canvas with violet and cyan accents instead of the previous flat warm-neutral palette.
- Create depth with restrained gradients, translucent surfaces, layered borders, and soft glows.
- Make the upload experience feel editorial and asymmetrical while preserving the existing workflow and English copy.
- Present extracted assets as a compact visual gallery with stronger thumbnails and clearer selected states.
- Keep the editable canvas as the dominant workspace surface.

## Component preview

- The dialog grows responsively up to the available viewport rather than imposing a fixed image height.
- The image keeps its intrinsic aspect ratio through `width: auto`, `height: auto`, `max-width: 100%`, and a viewport-aware `max-height`.
- The image stage centers the complete asset and uses a checkerboard for transparent layers.
- The only primary action remains `Download`.

## Constraints

- Do not change backend APIs or extraction behavior.
- Keep all visible product text in English.
- Preserve upload, automated/manual extraction, drag-and-drop canvas, ordering, download, and export behavior.
- Support desktop and mobile layouts without horizontal overflow.

## Verification

- Add a regression assertion that preview dimensions and the contain-fit hook are present.
- Run the full Vitest suite, ESLint, and Vite production build.
- Inspect upload, workspace, and component preview states at desktop and mobile widths.
