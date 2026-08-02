import { useEffect, useRef, useState } from 'react'

import { processImage } from '../api'
import { validateManualKeywords } from '../lib/keywords'
import ExtractionModeSelector from './ExtractionModeSelector'

const MAX_FILE_SIZE = 10 * 1024 * 1024
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp']

export default function UploadScreen({
  onProcessed,
  processRequest = processImage,
}) {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  const [mode, setMode] = useState('automated')
  const [keywordText, setKeywordText] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL?.(preview)
  }, [preview])

  const chooseFile = (nextFile) => {
    if (!nextFile) return
    if (!ALLOWED_TYPES.includes(nextFile.type)) {
      setError('Choose a JPEG, PNG, or WebP image.')
      return
    }
    if (nextFile.size > MAX_FILE_SIZE) {
      setError('Choose an image smaller than 10 MB.')
      return
    }

    setFile(nextFile)
    setError('')
    setPreview(URL.createObjectURL?.(nextFile) || '')
  }

  const submit = async () => {
    if (!file) {
      setError('Choose an image before extracting layers.')
      return
    }

    let keywords = null
    if (mode === 'manual') {
      const validation = validateManualKeywords(keywordText)
      if (validation.error) {
        setError(validation.error)
        return
      }
      keywords = validation.keywords
    }

    setLoading(true)
    setError('')
    try {
      onProcessed(await processRequest(file, keywords))
    } catch (requestError) {
      setError(requestError.message || 'Layer extraction failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="upload-page">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="Magic Layers home">
          <span className="brand-mark" aria-hidden="true">M</span>
          <span>Magic Layers</span>
        </a>
        <span className="header-note">AI-powered image decomposition</span>
      </header>

      <section className="upload-content" id="top">
        <div className="hero-copy">
          <p className="eyebrow">Image to editable layers</p>
          <h1>Turn one image into a canvas you can rearrange.</h1>
          <p className="hero-description">
            Upload an image, isolate every meaningful object, and compose the
            result freely on an editable canvas.
          </p>
        </div>

        <div
          className="dropzone"
          data-has-preview={Boolean(preview)}
          onDragOver={(event) => event.preventDefault()}
          onDrop={(event) => {
            event.preventDefault()
            chooseFile(event.dataTransfer.files[0])
          }}
        >
          {preview ? (
            <div className="selected-image">
              <img src={preview} alt="Selected upload preview" />
              <div className="selected-image-meta">
                <span><strong>{file.name}</strong><small>{(file.size / 1024 / 1024).toFixed(1)} MB</small></span>
                <button type="button" className="text-button" onClick={() => inputRef.current?.click()}>Replace image</button>
              </div>
            </div>
          ) : (
            <button type="button" className="dropzone-action" onClick={() => inputRef.current?.click()}>
              <span className="upload-glyph" aria-hidden="true">↑</span>
              <strong>Drop an image here or browse</strong>
              <small>JPEG, PNG or WebP · up to 10 MB</small>
            </button>
          )}
          <input
            ref={inputRef}
            className="visually-hidden"
            type="file"
            accept="image/jpeg,image/png,image/webp"
            aria-label="Choose image"
            onChange={(event) => chooseFile(event.target.files[0])}
          />
        </div>

        <ExtractionModeSelector mode={mode} onChange={(nextMode) => {
          setMode(nextMode)
          setError('')
        }} />

        {mode === 'manual' && (
          <div className="manual-panel">
            <label className="keyword-field">
              <span>Object keywords</span>
              <input
                value={keywordText}
                onChange={(event) => setKeywordText(event.target.value)}
                placeholder='dog, "wooden chair", plant'
              />
            </label>
            <div className="keyword-help">
              <strong>How to enter keywords</strong>
              <p>Separate keywords with commas. Wrap multi-word keywords in double quotes. Maximum 10 keywords.</p>
            </div>
          </div>
        )}

        <div className="submit-row">
          <div aria-live="polite">
            {error && <p className="form-error">{error}</p>}
          </div>
          <button type="button" className="primary-button" disabled={loading} onClick={submit}>
            {loading ? 'Extracting layers…' : 'Extract layers'}
            {!loading && <span aria-hidden="true">→</span>}
          </button>
        </div>
      </section>
    </main>
  )
}
