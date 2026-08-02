import { describe, expect, it } from 'vitest'

import { fitCanvasSize } from './canvas'

describe('fitCanvasSize', () => {
  it('fits a landscape image within both available dimensions', () => {
    expect(fitCanvasSize(1600, 1000, 800, 400)).toEqual({
      width: 640,
      height: 400,
      scale: 0.4,
    })
  })

  it('fits a portrait image by its height', () => {
    expect(fitCanvasSize(800, 1200, 700, 600)).toEqual({
      width: 400,
      height: 600,
      scale: 0.5,
    })
  })

  it('does not upscale an image beyond original dimensions', () => {
    expect(fitCanvasSize(400, 240, 1200, 900)).toEqual({
      width: 400,
      height: 240,
      scale: 1,
    })
  })
})
