# Magic Canvas Frontend Redesign

## Goal

Build a complete English-language frontend for uploading an image, choosing automated or manual layer extraction, and editing the returned background and isolated components on a draggable canvas.

The frontend must keep using the existing FastAPI contract without backend changes.

## Scope

- Redesign the existing React/Vite frontend.
- Preserve the existing `POST /api/process-image` endpoint and response fields.
- Preserve Fabric.js as the draggable canvas engine.
- Provide responsive upload and result screens.
- Add individual previews and downloads for every extracted component and the background.
- Do not add OpenAPI code generation, a mock server, authentication, persistence, or backend optimization in this iteration.

## Visual Direction

Use a refined editor aesthetic with warm neutral surfaces, restrained green accents, generous whitespace, clear hierarchy, and compact controls. The interface must feel like a focused creative tool rather than a generic dashboard.

All user-facing text, validation, status messages, labels, filenames, and error fallbacks must be in English.

## Upload Screen

The first screen contains:

1. Product identity and a short English explanation.
2. A drag-and-drop image upload area that also opens the system file picker.
3. A selected-image preview with a way to replace the image.
4. Two mutually exclusive extraction controls:
   - `Automated Layer Extraction`
   - `Manual Layer Extraction`
5. A primary action that starts extraction.

Automated mode is selected by default. In automated mode the frontend submits only the image file and omits the `keywords` form field, allowing the backend workflow to generate keywords.

Manual mode reveals a keyword input and adjacent guidance. The guidance explains that keywords are comma-separated, multi-word keywords must be wrapped in double quotes, and no more than 10 keywords are allowed. Example input: `dog, "wooden chair", plant`.

Manual submission requires at least one valid keyword and prevents submission when more than 10 comma-separated entries are supplied. The frontend parses the quoted input, removes the syntactic quote characters, and sends the normalized keywords as the comma-separated string expected by the existing backend.

## Loading and Error States

While processing, the primary action is disabled and communicates that extraction may take some time. Repeated submissions are prevented.

Upload validation and API failures appear near the primary action in English. The frontend maps HTTP status codes and FastAPI failures to stable English messages instead of exposing potentially non-English backend details.

## Result Workspace

The result screen uses a two-column editor layout:

- Left: component browser.
- Right: draggable Fabric.js canvas.

The header reports component count and original image dimensions and includes a way to return to the upload screen. Returning does not require a page reload.

### Component Browser

The browser lists the background first, followed by every entry in `layers`. Each row contains an isolated thumbnail, a readable label, and compact metadata. Component rows are generated from the response and remain separate even when multiple layers share a keyword.

Clicking any component or the background opens a large modal preview. The preview uses a checkerboard surface for transparent component PNGs. The background preview uses the complete background image. The modal includes:

- The item name.
- A large image preview.
- One button labeled exactly `Download`.
- A close control and backdrop dismissal.

Download filenames are sanitized but retain the item identity, for example `dog.png`, `wooden-chair.png`, and `background.png`. No keyword is appended to the visible button label.

### Canvas

The background and all components are loaded from the existing base64 response. The background stays behind component objects. Components remain selectable and draggable, with controls for stacking order. The canvas preserves the source aspect ratio and adapts to the available viewport.

Selecting an item on the canvas updates the selected state in the editor. The final composed canvas can be exported as a PNG at original-image resolution.

## Data Flow

1. The user selects a file and extraction mode.
2. The frontend validates the file and, in manual mode, the keyword count.
3. The API adapter creates `FormData` with `file` and conditionally `keywords`.
4. `POST /api/process-image` returns `background_base64`, original dimensions, and isolated layers.
5. The component browser derives downloadable items from the response.
6. Fabric.js composes those same response assets on the editable canvas.
7. Individual downloads use the original base64 assets; canvas export uses Fabric.js output.

## Component Boundaries

- `App`: owns the upload/result screen transition and processed result.
- `UploadScreen`: owns file selection, extraction-mode UI, validation feedback, and submission state.
- `ExtractionModeSelector`: owns the automated/manual choice and manual guidance presentation.
- `Workspace`: arranges the header, component browser, preview modal, and canvas.
- `ComponentBrowser`: maps the background and layers into individually selectable rows.
- `ComponentPreviewModal`: renders the enlarged asset and its `Download` action.
- `CanvasComponent`: owns Fabric.js lifecycle, object selection, ordering, dragging, and canvas export.
- `api`: is the only module that knows the backend base URL and multipart request details.
- `image`: contains base64 URL conversion, filename sanitization, and browser download helpers.

## Responsive Behavior

Desktop uses the full two-column workspace. On smaller screens, the component browser stacks above the canvas and component rows form a compact grid where space permits. The preview modal remains within the viewport and reduces image height without hiding its close or download controls.

## Testing

Frontend tests cover:

- Automated submissions omit `keywords`.
- Manual selection reveals keyword controls.
- Manual validation accepts quoted multi-word keywords and rejects more than 10 entries.
- Successful processing switches to the result workspace.
- Component rows include the background and every returned layer.
- Clicking a component opens its enlarged preview.
- The preview action is labeled exactly `Download` and uses the selected image.
- API and validation failures display an English fallback where applicable.

Static verification includes ESLint and a production Vite build. Canvas behavior receives focused unit coverage for deterministic helpers and manual browser verification for Fabric.js interactions and responsive layout.

## Acceptance Criteria

- The complete user-facing interface is English.
- Both extraction modes call the existing API with the correct multipart fields.
- Manual instructions match the quote, comma, and 10-keyword requirements.
- Every component and the background can be previewed individually at a larger size.
- Every preview contains one button labeled `Download`.
- Every extracted component remains a separate image and draggable canvas object.
- The canvas can export the composition as a PNG.
- The frontend builds and lints successfully on the dedicated branch.
