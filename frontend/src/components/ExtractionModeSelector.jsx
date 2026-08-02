export default function ExtractionModeSelector({ mode, onChange }) {
  return (
    <div className="mode-grid" aria-label="Extraction mode">
      <button
        type="button"
        className="mode-card"
        data-active={mode === 'automated'}
        aria-pressed={mode === 'automated'}
        onClick={() => onChange('automated')}
      >
        <span className="mode-icon" aria-hidden="true">✦</span>
        <span>
          <strong>Automated Layer Extraction</strong>
          <small>Let AI identify the important objects automatically.</small>
        </span>
      </button>

      <button
        type="button"
        className="mode-card"
        data-active={mode === 'manual'}
        aria-pressed={mode === 'manual'}
        onClick={() => onChange('manual')}
      >
        <span className="mode-icon" aria-hidden="true">⌁</span>
        <span>
          <strong>Manual Layer Extraction</strong>
          <small>Choose exactly which objects should become layers.</small>
        </span>
      </button>
    </div>
  )
}
