import { useEffect } from 'react'

import { downloadDataUrl, toImageDataUrl } from '../lib/image'

export default function ComponentPreviewModal({ item, onClose }) {
  useEffect(() => {
    const closeOnEscape = (event) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', closeOnEscape)
    return () => window.removeEventListener('keydown', closeOnEscape)
  }, [onClose])

  return (
    <div
      className="preview-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <section className="preview-dialog" role="dialog" aria-modal="true" aria-labelledby="preview-title">
        <header className="preview-header">
          <div>
            <p className="panel-eyebrow">{item.kind === 'background' ? 'Background' : 'Isolated component'}</p>
            <h2 id="preview-title">{item.name}</h2>
          </div>
          <button type="button" className="icon-button" aria-label="Close preview" onClick={onClose}>×</button>
        </header>

        <div className={`preview-image ${item.kind === 'layer' ? 'checkerboard' : ''}`}>
          <img
            className="preview-object"
            src={toImageDataUrl(item.image)}
            alt={`${item.name} preview`}
            width={item.width}
            height={item.height}
          />
        </div>

        <button
          type="button"
          className="download-button"
          onClick={() => downloadDataUrl(item.image, item.name)}
        >
          <span aria-hidden="true">↓</span>
          Download
        </button>
      </section>
    </div>
  )
}
