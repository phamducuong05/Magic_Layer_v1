// @vitest-environment node

import { describe, expect, it } from 'vitest'

import viteConfig from '../vite.config'

describe('LAN server configuration', () => {
  it('listens on every server interface at port 5173', () => {
    expect(viteConfig.server).toEqual({
      host: '0.0.0.0',
      port: 5173,
    })
  })
})
