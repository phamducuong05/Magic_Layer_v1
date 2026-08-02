import { act, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const fabricState = vi.hoisted(() => ({ observers: [] }))

vi.mock('fabric', () => {
  class Canvas {
    constructor(element) {
      this.element = element
      this.objects = []
      this.handlers = {}
      this.wrapper = document.createElement('div')
      element.parentNode.insertBefore(this.wrapper, element)
      this.wrapper.appendChild(element)
    }
    on(name, handler) { this.handlers[name] = handler }
    add(object) { this.objects.push(object) }
    renderAll() {}
    getObjects() { return this.objects }
    getActiveObject() { return null }
    dispose() {
      if (this.wrapper.parentNode) {
        this.wrapper.parentNode.insertBefore(this.element, this.wrapper)
        this.wrapper.remove()
      }
    }
  }

  class FabricImage {
    static async fromURL() {
      return {
        set() {},
        layerName: '',
      }
    }
  }

  return { Canvas, FabricImage }
})

import CanvasComponent from './CanvasComponent'

const data = {
  background_base64: 'YmFja2dyb3VuZA==',
  original_width: 800,
  original_height: 600,
  layers: [],
}

describe('CanvasComponent', () => {
  beforeEach(() => {
    fabricState.observers = []
    globalThis.ResizeObserver = class ResizeObserver {
      constructor(callback) {
        this.callback = callback
        fabricState.observers.push(this)
      }
      observe() {}
      disconnect() {}
    }
  })

  it('remains mounted when its host is resized after Fabric reparents the canvas', async () => {
    render(<CanvasComponent data={data} />)
    await waitFor(() => expect(screen.queryByRole('status')).not.toBeInTheDocument())

    await act(async () => {
      fabricState.observers[0].callback([
        { contentRect: { width: 620, height: 480 } },
      ])
    })

    expect(screen.getByLabelText('Editable image canvas')).toBeInTheDocument()
  })
})
