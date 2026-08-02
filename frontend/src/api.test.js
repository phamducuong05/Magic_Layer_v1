import { describe, expect, it, vi } from 'vitest'

import { processImage } from './api'

const response = {
  background_base64: 'background',
  original_width: 1200,
  original_height: 800,
  layers: [],
}

function imageFile() {
  return new File(['image'], 'scene.png', { type: 'image/png' })
}

describe('processImage', () => {
  it('omits keywords for automated extraction', async () => {
    const fetchImpl = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => response,
    })

    await processImage(imageFile(), null, fetchImpl)

    const [, request] = fetchImpl.mock.calls[0]
    expect(request.body.get('file')).toBeInstanceOf(File)
    expect(request.body.has('keywords')).toBe(false)
  })

  it('sends normalized keywords for manual extraction', async () => {
    const fetchImpl = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => response,
    })

    await processImage(imageFile(), ['dog', 'wooden chair'], fetchImpl)

    const [, request] = fetchImpl.mock.calls[0]
    expect(request.body.get('keywords')).toBe('dog,wooden chair')
  })

  it('returns a stable English error for backend failures', async () => {
    const fetchImpl = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({ detail: 'Dịch vụ không khả dụng.' }),
    })

    await expect(processImage(imageFile(), null, fetchImpl)).rejects.toThrow(
      'The extraction service is temporarily unavailable.',
    )
  })
})
