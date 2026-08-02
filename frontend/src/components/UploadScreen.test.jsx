import { render, screen } from '@testing-library/react'
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

    expect(request).toHaveBeenCalledWith(expect.any(File), null)
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

    expect(request).toHaveBeenCalledWith(expect.any(File), [
      'dog',
      'wooden chair',
      'plant',
    ])
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
})
