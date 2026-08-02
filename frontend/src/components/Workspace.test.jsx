import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import Workspace from './Workspace'

const data = {
  background_base64: 'YmFja2dyb3VuZA==',
  original_width: 1200,
  original_height: 800,
  layers: [
    {
      keyword: 'Dog',
      png_base64: 'ZG9n',
      x: 10,
      y: 20,
      width: 300,
      height: 240,
    },
    {
      keyword: 'Wooden chair',
      png_base64: 'Y2hhaXI=',
      x: 400,
      y: 220,
      width: 260,
      height: 360,
    },
  ],
}

function CanvasStub() {
  return <div aria-label="Editable canvas" />
}

describe('Workspace', () => {
  it('renders the background and every returned component separately', () => {
    render(<Workspace data={data} onBack={() => {}} CanvasView={CanvasStub} />)

    expect(screen.getByRole('button', { name: /Preview Background/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Preview Dog/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Preview Wooden chair/i })).toBeInTheDocument()
  })

  it('opens a large isolated preview with a generic Download action', async () => {
    const user = userEvent.setup()
    render(<Workspace data={data} onBack={() => {}} CanvasView={CanvasStub} />)

    await user.click(screen.getByRole('button', { name: /Preview Dog/i }))

    const dialog = screen.getByRole('dialog', { name: 'Dog' })
    expect(within(dialog).getByRole('img', { name: 'Dog preview' })).toHaveAttribute(
      'src',
      'data:image/png;base64,ZG9n',
    )
    expect(within(dialog).getByRole('button', { name: 'Download' })).toBeInTheDocument()
    expect(within(dialog).queryByRole('button', { name: 'Download Dog' })).not.toBeInTheDocument()
  })

  it('closes the preview and returns to the same workspace', async () => {
    const user = userEvent.setup()
    render(<Workspace data={data} onBack={() => {}} CanvasView={CanvasStub} />)

    await user.click(screen.getByRole('button', { name: /Preview Wooden chair/i }))
    await user.click(screen.getByRole('button', { name: 'Close preview' }))

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.getByLabelText('Editable canvas')).toBeInTheDocument()
  })

  it('returns to upload through the New image action', async () => {
    const user = userEvent.setup()
    const onBack = vi.fn()
    render(<Workspace data={data} onBack={onBack} CanvasView={CanvasStub} />)

    await user.click(screen.getByRole('button', { name: 'New image' }))

    expect(onBack).toHaveBeenCalledOnce()
  })
})
