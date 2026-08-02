export function fitCanvasSize(
  originalWidth,
  originalHeight,
  availableWidth,
  availableHeight,
) {
  const scale = Math.min(
    1,
    availableWidth / originalWidth,
    availableHeight / originalHeight,
  )

  return {
    width: Math.max(1, Math.round(originalWidth * scale)),
    height: Math.max(1, Math.round(originalHeight * scale)),
    scale,
  }
}
