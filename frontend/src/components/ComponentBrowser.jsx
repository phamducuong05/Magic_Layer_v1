import { toImageDataUrl } from '../lib/image'

export default function ComponentBrowser({ items, selectedId, onPreview }) {
  return (
    <aside className="component-panel">
      <div className="component-panel-heading">
        <div>
          <p className="panel-eyebrow">Asset library</p>
          <h2>Components</h2>
        </div>
        <span>{items.length} items</span>
      </div>

      <p className="component-hint">Select any item to inspect and download it.</p>

      <div className="component-list">
        {items.map((item) => (
          <button
            type="button"
            className="component-row"
            data-selected={item.id === selectedId}
            aria-label={`Preview ${item.name}`}
            key={item.id}
            onClick={() => onPreview(item)}
          >
            <span className={`component-thumbnail ${item.kind === 'layer' ? 'checkerboard' : ''}`}>
              <img src={toImageDataUrl(item.image)} alt="" />
            </span>
            <span className="component-copy">
              <strong>{item.name}</strong>
              <small>
                {item.kind === 'background'
                  ? `${item.width} × ${item.height}px`
                  : `${item.width} × ${item.height}px · PNG`}
              </small>
            </span>
            <span className="row-arrow" aria-hidden="true">↗</span>
          </button>
        ))}
      </div>
    </aside>
  )
}
