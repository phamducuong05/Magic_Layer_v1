import { describe, expect, it } from 'vitest'

import { parseKeywords, validateManualKeywords } from './keywords'

describe('parseKeywords', () => {
  it('removes syntactic quotes while preserving multi-word keywords', () => {
    expect(parseKeywords('dog, "wooden chair", plant')).toEqual([
      'dog',
      'wooden chair',
      'plant',
    ])
  })

  it('keeps commas inside a quoted keyword', () => {
    expect(parseKeywords('"red, white flag", person')).toEqual([
      'red, white flag',
      'person',
    ])
  })
})

describe('validateManualKeywords', () => {
  it('rejects an empty manual keyword list', () => {
    expect(validateManualKeywords('')).toEqual({
      keywords: [],
      error: 'Enter at least one keyword.',
    })
  })

  it('rejects more than ten keywords', () => {
    expect(validateManualKeywords('a,b,c,d,e,f,g,h,i,j,k').error).toBe(
      'Use no more than 10 keywords.',
    )
  })

  it('accepts ten keywords', () => {
    expect(validateManualKeywords('a,b,c,d,e,f,g,h,i,j').error).toBe('')
  })
})
