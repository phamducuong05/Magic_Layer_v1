export function parseKeywords(value) {
  const keywords = []
  let current = ''
  let quoted = false

  for (const character of value) {
    if (character === '"') {
      quoted = !quoted
      continue
    }

    if (character === ',' && !quoted) {
      const keyword = current.trim()
      if (keyword) keywords.push(keyword)
      current = ''
      continue
    }

    current += character
  }

  const keyword = current.trim()
  if (keyword) keywords.push(keyword)
  return keywords
}

export function validateManualKeywords(value) {
  const keywords = parseKeywords(value)

  if (keywords.length === 0) {
    return { keywords, error: 'Enter at least one keyword.' }
  }

  if (keywords.length > 10) {
    return { keywords, error: 'Use no more than 10 keywords.' }
  }

  return { keywords, error: '' }
}
