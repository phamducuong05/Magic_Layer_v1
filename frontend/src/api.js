const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Gửi ảnh + keywords đến backend, nhận về layers + background.
 * @param {File} file - File ảnh từ input
 * @param {string} keywords - Chuỗi keywords cách nhau bởi dấu phẩy
 * @returns {Promise<ProcessResponse>}
 */
export async function processImage(file, keywords) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('keywords', keywords)

  const res = await fetch(`${API_BASE}/api/process-image`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(err.detail || `HTTP ${res.status}`)
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