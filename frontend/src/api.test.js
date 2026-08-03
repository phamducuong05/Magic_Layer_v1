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

function jsonResponse(body, { ok = true, status = 200 } = {}) {
  return { ok, status, json: async () => body }
}

describe('processImage', () => {
  it('creates an automated job without keywords and returns its result', async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        job_id: 'job-1', status: 'queued', stage: 'queued', progress: 0,
        message: 'Waiting to start',
      }, { status: 202 }))
      .mockResolvedValueOnce(jsonResponse({
        job_id: 'job-1', status: 'completed', stage: 'completed', progress: 100,
        message: 'Layers are ready', result: response,
      }))

    const result = await processImage(
      imageFile(), null, vi.fn(), fetchImpl, async () => {},
    )

    const [url, request] = fetchImpl.mock.calls[0]
    expect(url).toBe(
      'http://192.168.1.20:8009/api/process-image/jobs',
    )
    expect(request.body.get('file')).toBeInstanceOf(File)
    expect(request.body.has('keywords')).toBe(false)
    expect(fetchImpl.mock.calls[1][0]).toMatch(/\/jobs\/job-1$/)
    expect(result).toEqual(response)
  })

  it('sends manual keywords and publishes real progress snapshots', async () => {
    const onProgress = vi.fn()
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        job_id: 'job-2', status: 'queued', stage: 'queued', progress: 0,
        message: 'Waiting to start',
      }, { status: 202 }))
      .mockResolvedValueOnce(jsonResponse({
        job_id: 'job-2', status: 'processing', stage: 'segmentation', progress: 15,
        message: 'Finding image components',
      }))
      .mockResolvedValueOnce(jsonResponse({
        job_id: 'job-2', status: 'completed', stage: 'completed', progress: 100,
        message: 'Layers are ready', result: response,
      }))

    await processImage(
      imageFile(), ['dog', 'wooden chair'], onProgress, fetchImpl, async () => {},
    )

    expect(fetchImpl.mock.calls[0][1].body.get('keywords')).toBe('dog,wooden chair')
    expect(onProgress).toHaveBeenCalledWith({
      stage: 'segmentation', progress: 15, message: 'Finding image components',
    })
    expect(onProgress).toHaveBeenLastCalledWith({
      stage: 'completed', progress: 100, message: 'Layers are ready',
    })
  })

  it('uses the job error when background processing fails', async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        job_id: 'job-3', status: 'queued', stage: 'queued', progress: 0,
        message: 'Waiting to start',
      }, { status: 202 }))
      .mockResolvedValueOnce(jsonResponse({
        job_id: 'job-3', status: 'failed', stage: 'failed', progress: 35,
        message: 'Processing failed', error: 'The image analysis service failed.',
      }))

    await expect(processImage(
      imageFile(), null, vi.fn(), fetchImpl, async () => {},
    )).rejects.toThrow('The image analysis service failed.')
  })

  it('rejects a completed job that has no result payload', async () => {
    const fetchImpl = vi.fn()
      .mockResolvedValueOnce(jsonResponse({
        job_id: 'job-4', status: 'completed', stage: 'completed', progress: 100,
        message: 'Layers are ready', result: null,
      }, { status: 202 }))

    await expect(processImage(
      imageFile(), null, vi.fn(), fetchImpl, async () => {},
    )).rejects.toThrow('The server returned an invalid processing result.')
  })

  it('returns a stable English error when job creation fails', async () => {
    const fetchImpl = vi.fn().mockResolvedValue(jsonResponse(
      { detail: 'Unavailable' }, { ok: false, status: 503 },
    ))

    await expect(processImage(
      imageFile(), null, vi.fn(), fetchImpl, async () => {},
    )).rejects.toThrow('The extraction service is temporarily unavailable.')
  })
})
