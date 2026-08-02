const STEPS = [
  ['keywords', 'Keywords'],
  ['segmentation', 'Components'],
  ['overlap', 'Overlaps'],
  ['reconstruction', 'Reconstruction'],
  ['grouping', 'Grouping'],
  ['layers', 'Layers'],
  ['background', 'Background'],
  ['completed', 'Ready'],
]

export default function ProcessingProgress({ progress }) {
  const value = Math.max(0, Math.min(100, progress?.progress || 0))
  const reportedStage = progress?.stage || 'keywords'
  const activeStage = reportedStage === 'complete'
    ? 'completed'
    : reportedStage === 'queued' ? 'keywords' : reportedStage
  const activeIndex = STEPS.findIndex(([stage]) => stage === activeStage)

  return (
    <section className="processing-progress" aria-labelledby="processing-title">
      <div className="processing-progress-header">
        <div>
          <p className="eyebrow">Live processing</p>
          <h2 id="processing-title">Building your editable layers</h2>
        </div>
        <strong className="progress-value">{value}%</strong>
      </div>

      <div
        className="progress-track"
        role="progressbar"
        aria-label="Layer extraction progress"
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={value}
      >
        <span style={{ width: `${value}%` }} />
      </div>

      <p className="progress-message" aria-live="polite">
        <span className="progress-pulse" aria-hidden="true" />
        {progress?.message || 'Preparing extraction'}
      </p>

      <ol className="progress-steps">
        {STEPS.map(([stage, label], index) => {
          const state = index < activeIndex
            ? 'complete'
            : index === activeIndex ? 'active' : 'pending'
          return (
            <li key={stage} data-state={state}>
              <span aria-hidden="true">{state === 'complete' ? '✓' : index + 1}</span>
              <small>{label}</small>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
