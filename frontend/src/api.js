const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8009'

const STATUS_MESSAGES = {
  400: 'Check the selected image and keywords, then try again.',
  502: 'The image analysis service returned an invalid response.',
  503: 'The extraction service is temporarily unavailable.',
}
const POLL_INTERVAL_MS = 750
const MAX_POLLS = 960
const wait = (milliseconds) => new Promise((resolve) => {
  window.setTimeout(resolve, milliseconds)
})

function publishProgress(job, onProgress) {
  onProgress?.({
    stage: job.stage,
    progress: job.progress,
    message: job.message,
  })
}

function requestError(status) {
  return new Error(
    STATUS_MESSAGES[status] || 'Layer extraction failed. Please try again.',
  )
}

/**
 * Gửi ảnh + keywords đến backend, nhận về layers + background.
 * @param {File} file - File ảnh từ input
 * @param {string[] | null} keywords - Normalized manual keywords, or null for automation
 * @param {(progress: object) => void} onProgress - Receives backend checkpoints
 * @param {typeof fetch} fetchImpl - Injectable fetch boundary for testing
 * @param {(milliseconds: number) => Promise<void>} waitImpl - Injectable delay
 * @returns {Promise<ProcessResponse>}
 */
export async function processImage(
  file,
  keywords = null,
  onProgress = () => {},
  fetchImpl = fetch,
  waitImpl = wait,
) {
  const formData = new FormData()
  formData.append('file', file)
  if (keywords?.length) {
    formData.append('keywords', keywords.join(','))
  }

  const res = await fetchImpl(`${API_BASE}/api/process-image/jobs`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    throw requestError(res.status)
  }

  let job = await res.json()
  if (!job.job_id) throw new Error('The server returned an invalid processing job.')
  publishProgress(job, onProgress)

  for (let pollCount = 0; pollCount < MAX_POLLS; pollCount += 1) {
    if (job.status === 'completed') {
      if (!job.result) {
        throw new Error('The server returned an invalid processing result.')
      }
      return job.result
    }
    if (job.status === 'failed') {
      throw new Error(job.error || 'Layer extraction failed. Please try again.')
    }

    await waitImpl(POLL_INTERVAL_MS)
    const statusResponse = await fetchImpl(
      `${API_BASE}/api/process-image/jobs/${encodeURIComponent(job.job_id)}`,
    )
    if (!statusResponse.ok) throw requestError(statusResponse.status)
    job = await statusResponse.json()
    publishProgress(job, onProgress)
  }

  throw new Error('Layer extraction timed out. Please try again.')
}

/**
 * Kiểm tra backend có sẵn sàng không.
 */
export async function healthCheck() {
  const res = await fetch(`${API_BASE}/health`)
  return res.ok
}
