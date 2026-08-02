import { useEffect, useRef, useState } from 'react'
import * as fabric from 'fabric'

import { fitCanvasSize } from './lib/canvas'
import { toImageDataUrl } from './lib/image'

export default function CanvasComponent({ data }) {
  const hostRef = useRef(null)
  const canvasElementRef = useRef(null)
  const fabricRef = useRef(null)
  const [hostSize, setHostSize] = useState({ width: 820, height: 560 })
  const [selectedName, setSelectedName] = useState('')
  const [ready, setReady] = useState(false)

  const { width, height, scale } = fitCanvasSize(
    data.original_width,
    data.original_height,
    Math.max(280, hostSize.width - 40),
    Math.max(240, hostSize.height - 40),
  )

  useEffect(() => {
    const host = hostRef.current
    if (!host) return undefined

    const updateSize = (nextWidth, nextHeight) => {
      if (nextWidth > 0 && nextHeight > 0) {
        setHostSize({ width: nextWidth, height: nextHeight })
      }
    }

    updateSize(host.clientWidth, host.clientHeight)
    const observer = new ResizeObserver((entries) => {
      const { width: nextWidth, height: nextHeight } = entries[0].contentRect
      updateSize(nextWidth, nextHeight)
    })
    observer.observe(host)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const canvasElement = canvasElementRef.current
    if (!canvasElement) return undefined

    let cancelled = false
    setReady(false)
    setSelectedName('')

    const canvas = new fabric.Canvas(canvasElement, {
      width,
      height,
      selection: true,
      preserveObjectStacking: true,
    })
    fabricRef.current = canvas

    const readSelection = (event) => {
      setSelectedName(event.selected?.[0]?.layerName || '')
    }
    canvas.on('selection:created', readSelection)
    canvas.on('selection:updated', readSelection)
    canvas.on('selection:cleared', () => setSelectedName(''))

    const load = async () => {
      try {
        const background = await fabric.FabricImage.fromURL(
          toImageDataUrl(data.background_base64),
          { crossOrigin: 'anonymous' },
        )
        if (cancelled) return
        background.set({
          left: 0,
          top: 0,
          originX: 'left',
          originY: 'top',
          scaleX: scale,
          scaleY: scale,
          selectable: false,
          evented: false,
        })
        background.layerName = 'Background'
        canvas.add(background)

        for (const [index, layer] of data.layers.entries()) {
          const image = await fabric.FabricImage.fromURL(
            toImageDataUrl(layer.png_base64),
            { crossOrigin: 'anonymous' },
          )
          if (cancelled) return
          image.set({
            left: layer.x * scale,
            top: layer.y * scale,
            originX: 'left',
            originY: 'top',
            scaleX: scale,
            scaleY: scale,
            borderColor: '#665dce',
            cornerColor: '#665dce',
            cornerStrokeColor: '#ffffff',
            cornerSize: 10,
            transparentCorners: false,
          })
          image.layerName = layer.keyword || `Component ${index + 1}`
          canvas.add(image)
        }

        canvas.renderAll()
        setReady(true)
      } catch {
        if (!cancelled) setReady(true)
      }
    }

    load()
    return () => {
      cancelled = true
      canvas.dispose()
      fabricRef.current = null
    }
  }, [data, height, scale, width])

  const moveSelected = (direction) => {
    const canvas = fabricRef.current
    const activeObject = canvas?.getActiveObject()
    if (!activeObject) return
    if (direction === 'front') canvas.bringObjectForward(activeObject)
    if (direction === 'back') canvas.sendObjectBackwards(activeObject)
    const background = canvas.getObjects().find((object) => object.layerName === 'Background')
    if (background) canvas.sendObjectToBack(background)
    canvas.renderAll()
  }

  const exportCanvas = () => {
    const dataUrl = fabricRef.current?.toDataURL({
      format: 'png',
      multiplier: 1 / scale,
    })
    if (!dataUrl) return
    const link = document.createElement('a')
    link.href = dataUrl
    link.download = 'magic-layers-composition.png'
    link.click()
  }

  return (
    <div className="canvas-editor">
      <div className="canvas-toolbar">
        <div className="canvas-selection">
          <span className="selection-dot" data-active={Boolean(selectedName)} />
          <span>{selectedName || 'Select a component'}</span>
        </div>
        <div className="toolbar-actions">
          <button type="button" onClick={() => moveSelected('front')} disabled={!selectedName}>Bring forward</button>
          <button type="button" onClick={() => moveSelected('back')} disabled={!selectedName}>Send backward</button>
          <button type="button" className="export-button" onClick={exportCanvas}>Export canvas <span aria-hidden="true">↓</span></button>
        </div>
      </div>

      <div className="canvas-surface" ref={hostRef}>
        <div className="canvas-frame" style={{ width, height }}>
          <div
            className="canvas-loading"
            data-visible={!ready}
            role={ready ? undefined : 'status'}
            aria-hidden={ready}
          >
            <span className="spinner" />
            <span>Preparing your canvas…</span>
          </div>
          <canvas ref={canvasElementRef} aria-label="Editable image canvas" />
        </div>
      </div>
      <p className="canvas-dimensions">{width} × {height}px · {Math.round(scale * 100)}% view</p>
    </div>
  )
}
