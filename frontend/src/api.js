const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8009'

const STATUS_MESSAGES = {
  400: 'Check the selected image and keywords, then try again.',
  502: 'The image analysis service returned an invalid response.',
  503: 'The extraction service is temporarily unavailable.',
}

/**
 * Gửi ảnh + keywords đến backend, nhận về layers + background.
 * @param {File} file - File ảnh từ input
 * @param {string[] | null} keywords - Normalized manual keywords, or null for automation
 * @param {typeof fetch} fetchImpl - Injectable fetch boundary for testing
 * @returns {Promise<ProcessResponse>}
 */
export async function processImage(file, keywords = null, fetchImpl = fetch) {
  const formData = new FormData()
  formData.append('file', file)
  if (keywords?.length) {
    formData.append('keywords', keywords.join(','))
  }

  const res = await fetchImpl(`${API_BASE}/api/process-image`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    throw new Error(
      STATUS_MESSAGES[res.status] || 'Layer extraction failed. Please try again.',
    )
  }

  return res.json()
}

/**
 * Kiểm tra backend có sẵn sàng không.
 */
export async function healthCheck() {
  const res = await fetch(`${API_BASE}/health`)
  return res.ok
}
