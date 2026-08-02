import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import UploadScreen from './UploadScreen'

const result = {
  background_base64: 'background',
  original_width: 1200,
  original_height: 800,
  layers: [],
}

function imageFile() {
  return new File(['image'], 'scene.png', { type: 'image/png' })
}

describe('UploadScreen', () => {
  it('shows manual keyword guidance only after manual mode is selected', async () => {
    const user = userEvent.setup()
    render(<UploadScreen onProcessed={() => {}} />)

    expect(screen.queryByLabelText('Object keywords')).not.toBeInTheDocument()

    await user.click(
      screen.getByRole('button', { name: /Manual Layer Extraction/i }),
    )

    expect(screen.getByLabelText('Object keywords')).toBeInTheDocument()
    expect(screen.getByText(/Wrap multi-word keywords in double quotes/i)).toBeInTheDocument()
  })

  it('submits automated extraction without keywords', async () => {
    const user = userEvent.setup()
    const request = vi.fn().mockResolvedValue(result)
    const onProcessed = vi.fn()
    render(<UploadScreen onProcessed={onProcessed} processRequest={request} />)

    await user.upload(screen.getByLabelText('Choose image'), imageFile())
    await user.click(screen.getByRole('button', { name: 'Extract layers' }))

    expect(request).toHaveBeenCalledWith(
      expect.any(File), null, expect.any(Function),
    )
    expect(onProcessed).toHaveBeenCalledWith(result)
  })

  it('normalizes quoted keywords for manual extraction', async () => {
    const user = userEvent.setup()
    const request = vi.fn().mockResolvedValue(result)
    render(<UploadScreen onProcessed={() => {}} processRequest={request} />)

    await user.upload(screen.getByLabelText('Choose image'), imageFile())
    await user.click(screen.getByRole('button', { name: /Manual Layer Extraction/i }))
    await user.type(
      screen.getByLabelText('Object keywords'),
      'dog, "wooden chair", plant',
    )
    await user.click(screen.getByRole('button', { name: 'Extract layers' }))

    expect(request).toHaveBeenCalledWith(
      expect.any(File),
      ['dog', 'wooden chair', 'plant'],
      expect.any(Function),
    )
  })

  it('prevents manual submission with more than ten keywords', async () => {
    const user = userEvent.setup()
    const request = vi.fn()
    render(<UploadScreen onProcessed={() => {}} processRequest={request} />)

    await user.upload(screen.getByLabelText('Choose image'), imageFile())
    await user.click(screen.getByRole('button', { name: /Manual Layer Extraction/i }))
    await user.type(
      screen.getByLabelText('Object keywords'),
      'a,b,c,d,e,f,g,h,i,j,k',
    )
    await user.click(screen.getByRole('button', { name: 'Extract layers' }))

    expect(screen.getByText('Use no more than 10 keywords.')).toBeInTheDocument()
    expect(request).not.toHaveBeenCalled()
  })

  it('shows live backend progress while extraction is running', async () => {
    const user = userEvent.setup()
    let finishRequest
    const request = vi.fn((file, keywords, onProgress) => {
      onProgress({
        stage: 'overlap',
        progress: 35,
        message: 'Checking component overlaps',
      })
      return new Promise((resolve) => { finishRequest = resolve })
    })
    const onProcessed = vi.fn()
    render(<UploadScreen onProcessed={onProcessed} processRequest={request} />)

    await user.upload(screen.getByLabelText('Choose image'), imageFile())
    await user.click(screen.getByRole('button', { name: 'Extract layers' }))

    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '35')
    expect(screen.getByText('Checking component overlaps')).toBeInTheDocument()
    expect(screen.getByText('35%')).toBeInTheDocument()

    finishRequest(result)
    await waitFor(() => expect(onProcessed).toHaveBeenCalledWith(result))
  })
})
