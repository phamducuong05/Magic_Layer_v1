export function toImageDataUrl(value) {
  if (!value) return ''
  return value.startsWith('data:') ? value : `data:image/png;base64,${value}`
}

export function sanitizeFilename(value) {
  const normalized = value
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-|-$/g, '')

  return normalized || 'component'
}

export function downloadDataUrl(value, name) {
  const link = document.createElement('a')
  link.href = toImageDataUrl(value)
  link.download = `${sanitizeFilename(name)}.png`
  document.body.appendChild(link)
  link.click()
  link.remove()
}
