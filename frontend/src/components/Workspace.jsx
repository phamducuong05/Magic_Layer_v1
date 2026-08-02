import { useMemo, useState } from 'react'

import CanvasComponent from '../CanvasComponent'
import ComponentBrowser from './ComponentBrowser'
import ComponentPreviewModal from './ComponentPreviewModal'

export default function Workspace({ data, onBack, CanvasView = CanvasComponent }) {
  const [previewItem, setPreviewItem] = useState(null)
  const items = useMemo(() => [
    {
      id: 'background',
      kind: 'background',
      name: 'Background',
      image: data.background_base64,
      width: data.original_width,
      height: data.original_height,
    },
    ...data.layers.map((layer, index) => ({
      id: `layer-${index}`,
      kind: 'layer',
      name: layer.keyword || `Component ${index + 1}`,
      image: layer.png_base64,
      width: layer.width,
      height: layer.height,
      layer,
    })),
  ], [data])

  return (
    <main className="workspace-page">
      <header className="workspace-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">M</span>
          <span>Magic Layers</span>
        </div>
        <div className="workspace-meta">
          <span>{data.layers.length} components</span>
          <span aria-hidden="true">·</span>
          <span>{data.original_width} × {data.original_height}px</span>
        </div>
        <button type="button" className="secondary-button" onClick={onBack}><span aria-hidden="true">←</span> New image</button>
      </header>

      <div className="workspace-layout">
        <ComponentBrowser
          items={items}
          selectedId={previewItem?.id}
          onPreview={setPreviewItem}
        />
        <section className="canvas-panel">
          <CanvasView data={data} />
        </section>
      </div>

      {previewItem && (
        <ComponentPreviewModal item={previewItem} onClose={() => setPreviewItem(null)} />
      )}
    </main>
  )
}
