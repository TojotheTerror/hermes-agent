import { describe, expect, it, vi } from 'vitest'

import {
  dedupeGeneratedImageEchoesInParts,
  generatedImageEchoSources,
  generatedImageFromResult,
  stripGeneratedImageEchoes
} from './generated-images'

describe('generatedImageFromResult', () => {
  it('prefers the host-visible image path', () => {
    expect(
      generatedImageFromResult({
        agent_visible_image: '/container/cache/cat.png',
        host_image: '/Users/me/.hermes/cache/images/cat.png',
        image: '/Users/me/.hermes/cache/images/cat.png',
        success: true
      })
    ).toBe('/Users/me/.hermes/cache/images/cat.png')
  })

  it('ignores failed image generation results', () => {
    expect(generatedImageFromResult({ image: 'https://cdn.example/cat.png', success: false })).toBeNull()
  })
})

describe('stripGeneratedImageEchoes', () => {
  it('removes repeated generated image markdown without removing prose', () => {
    expect(
      stripGeneratedImageEchoes('Here you go.\n\n![Generated image](https://cdn.example/cat.png)', [
        'https://cdn.example/cat.png'
      ])
    ).toBe('Here you go.\n\n')
  })

  it('removes media links for generated local image paths', () => {
    expect(stripGeneratedImageEchoes('Saved image: [Image: cat.png](#media:%2Ftmp%2Fcat.png)', ['/tmp/cat.png'])).toBe(
      'Saved image: '
    )
  })

  it('keeps a MEDIA: attachment link whose path is NOT a generated source (#80621)', () => {
    // The reporter's case A: a final response with prose + a MEDIA: directive,
    // in a turn that also completed an image_generate tool call for a DIFFERENT
    // file. The attachment link must survive the echo-dedupe.
    const text =
      'Here is the rendered proof:\n[Image: render-proof.jpg](#media:%2Ftmp%2Frender-proof.jpg)\nIt should show the layout correctly.'

    expect(stripGeneratedImageEchoes(text, ['/tmp/other-generated.png'])).toBe(text)
  })

  it('keeps a MEDIA: directive for a different file while stripping the same-file echo (#80621)', () => {
    const text =
      'Here is the rendered proof:\n[Image: render-proof.jpg](#media:%2Ftmp%2Frender-proof.jpg)\nAnd the raw output: ![Generated](/tmp/generated.png) done.'

    // Two spaces where the echo stood: main stopped normalizing whole-document
    // whitespace through media extraction (3e1b7af21be — it erased hard breaks
    // and fenced-code spacing), so the strip leaves the gap as-is.
    expect(stripGeneratedImageEchoes(text, ['/tmp/generated.png'])).toBe(
      'Here is the rendered proof:\n[Image: render-proof.jpg](#media:%2Ftmp%2Frender-proof.jpg)\nAnd the raw output:  done.'
    )
  })

  it('keeps backticked absolute paths inert (#80621 case B)', () => {
    const text = 'The proof is at `/tmp/render-proof.png` and the other at `/tmp/render-proof-2.png`.'

    expect(stripGeneratedImageEchoes(text, ['/tmp/render-proof.png'])).toBe(text)
  })

  it('does not over-match a longer path that merely starts with the source', () => {
    // `/tmp/a.png` must not eat a link to `/tmp/a.png.bak` — prefix over-match
    // would silently drop a legitimate attachment.
    const text = 'Keep this: [Image: a.png.bak](#media:%2Ftmp%2Fa.png.bak)'

    expect(stripGeneratedImageEchoes(text, ['/tmp/a.png'])).toBe(text)
  })

  it('does not corrupt a markdown link whose destination is the source path', () => {
    // A bare-path strip inside `[label](/tmp/a.png)` would leave a broken
    // `[label]()` link; the link must survive intact.
    const text = 'Open [the file](/tmp/a.png) to view it.'

    expect(stripGeneratedImageEchoes(text, ['/tmp/a.png'])).toBe(text)
  })
})

describe('generatedImageEchoSources', () => {
  it('collects every path variant the model might restate', () => {
    expect(
      generatedImageEchoSources([
        {
          result: {
            agent_visible_image: '/sandbox/cat.png',
            host_image: '/host/cat.png',
            image: '/host/cat.png',
            success: true
          },
          toolName: 'image_generate',
          type: 'tool-call'
        }
      ])
    ).toEqual(['/host/cat.png', '/sandbox/cat.png'])
  })
})

describe('dedupeGeneratedImageEchoesInParts', () => {
  it('preserves timeline identity when there is nothing to strip', () => {
    const parts = [{ result: {}, toolName: 'read_file', type: 'tool-call' }]

    expect(dedupeGeneratedImageEchoesInParts(parts)).toBe(parts)
  })

  it('keeps the agent prose while removing the duplicated image', () => {
    expect(
      dedupeGeneratedImageEchoesInParts([
        { text: 'Here is your peacock! ![peacock](/host/p.png) Enjoy.', type: 'text' },
        {
          result: { host_image: '/host/p.png', image: '/host/p.png', success: true },
          toolName: 'image_generate',
          type: 'tool-call'
        }
      ])
    ).toEqual([
      { text: 'Here is your peacock!  Enjoy.', type: 'text' },
      {
        result: { host_image: '/host/p.png', image: '/host/p.png', success: true },
        toolName: 'image_generate',
        type: 'tool-call'
      }
    ])
  })

  it('strips a sandbox path the model restated instead of the host path', () => {
    expect(
      dedupeGeneratedImageEchoesInParts([
        { text: '![cat](/sandbox/cat.png)', type: 'text' },
        {
          result: {
            agent_visible_image: '/sandbox/cat.png',
            host_image: '/host/cat.png',
            image: '/host/cat.png',
            success: true
          },
          toolName: 'image_generate',
          type: 'tool-call'
        }
      ])
    ).toEqual([
      {
        result: {
          agent_visible_image: '/sandbox/cat.png',
          host_image: '/host/cat.png',
          image: '/host/cat.png',
          success: true
        },
        toolName: 'image_generate',
        type: 'tool-call'
      }
    ])
  })

  it('leaves pending generations untouched so the agent prose survives', () => {
    const parts = [
      { text: 'Another peacock, coming up!', type: 'text' },
      { result: undefined, toolName: 'image_generate', type: 'tool-call' }
    ]

    expect(dedupeGeneratedImageEchoesInParts(parts)).toEqual(parts)
  })

  it('keeps the MEDIA: attachment link when it references a different file than the generation (#80621)', () => {
    const parts = [
      {
        text: 'Here is the rendered proof:\n[Image: render-proof.jpg](#media:%2Ftmp%2Frender-proof.jpg)\nIt should show the layout correctly.',
        type: 'text' as const
      },
      {
        result: { host_image: '/tmp/other-generated.png', image: '/tmp/other-generated.png', success: true },
        toolName: 'image_generate',
        type: 'tool-call' as const
      }
    ]

    expect(dedupeGeneratedImageEchoesInParts(parts)).toEqual(parts)
  })

  it('does not drop a message whose only content was a MEDIA: link to a different file (#80621)', () => {
    const parts = [
      { text: '[Image: render-proof.jpg](#media:%2Ftmp%2Frender-proof.jpg)', type: 'text' as const },
      {
        result: { host_image: '/tmp/other-generated.png', image: '/tmp/other-generated.png', success: true },
        toolName: 'image_generate',
        type: 'tool-call' as const
      }
    ]

    expect(dedupeGeneratedImageEchoesInParts(parts)).toEqual(parts)
  })
})

describe('generated image echoes with an inline data: URL', () => {
  // A data: URL this long made `new RegExp` throw "Regular expression too large" on every streamed
  // delta once an image_generate result carried it, flooding desktop.log.
  const dataUrl = `data:image/png;base64,${'iVBORw0KGgoAAAANSUhEUg'.repeat(20_000)}`

  it('does not throw and leaves prose without the URL untouched', () => {
    const text = 'Here is your peacock, rendered in watercolour.'

    expect(stripGeneratedImageEchoes(text, [dataUrl])).toBe(text)
  })

  it('does not encode the source for streamed text that cannot contain it', () => {
    const encode = vi.spyOn(globalThis, 'encodeURIComponent')

    try {
      stripGeneratedImageEchoes('A short streamed delta.', [dataUrl])
      stripGeneratedImageEchoes(`${'Long prose without a media link. '.repeat(20_000)}`, [dataUrl])
      expect(encode).not.toHaveBeenCalledWith(dataUrl)
    } finally {
      encode.mockRestore()
    }
  })

  it('removes a markdown image of the data: URL and keeps the prose around it', () => {
    expect(stripGeneratedImageEchoes(`Here you go.\n\n![Generated image](${dataUrl})\n\nEnjoy!`, [dataUrl])).toBe(
      'Here you go.\n\n\n\nEnjoy!'
    )
  })

  it('removes a markdown image wrapped in angle brackets with a title', () => {
    expect(stripGeneratedImageEchoes(`A ![cat](<${dataUrl}> "title") B`, [dataUrl])).toBe('A  B')
  })

  it('removes a media link to the data: URL, matching lowercase hex too', () => {
    const encoded = encodeURIComponent(dataUrl).replace(/%[0-9A-F]{2}/g, hex => hex.toLowerCase())

    expect(stripGeneratedImageEchoes(`Saved: [Image](#media:${encoded} )`, [dataUrl])).toBe('Saved: ')
  })

  it('removes a bare occurrence and keeps the boundary character', () => {
    expect(stripGeneratedImageEchoes(`Image: ${dataUrl}. Done`, [dataUrl])).toBe('Image: . Done')
    expect(stripGeneratedImageEchoes(`${dataUrl} first`, [dataUrl])).toBe(' first')
    expect(stripGeneratedImageEchoes(`[<${dataUrl}>]`, [dataUrl])).toBe('[]')
  })

  it('keeps a longer URL that merely starts with the generated one', () => {
    const longer = `${dataUrl}AAAA`

    expect(stripGeneratedImageEchoes(`![x](${longer}) and ${longer} stay`, [dataUrl])).toBe(
      `![x](${longer}) and ${longer} stay`
    )
  })

  it('keeps a markdown link destination intact', () => {
    expect(stripGeneratedImageEchoes(`[label](${dataUrl})`, [dataUrl])).toBe(`[label](${dataUrl})`)
  })

  it('dedupes parts whose image_generate result carries the data: URL without throwing', () => {
    const parts = [
      { text: `Done!\n\n![Generated image](${dataUrl})`, type: 'text' as const },
      { result: { image: dataUrl, success: true }, toolName: 'image_generate', type: 'tool-call' as const }
    ]

    expect(dedupeGeneratedImageEchoesInParts(parts)).toEqual([{ ...parts[0], text: 'Done!\n\n' }, parts[1]])
  })

  it('matches the RegExp path for a short source on the same inputs', () => {
    const short = '/tmp/cat.png'
    const long = `/tmp/${'x'.repeat(3000)}.png`

    const cases = (src: string) => [
      `![a](${src})`,
      `![a](<${src}>)`,
      `x ${src} y`,
      `{${src}}`,
      `[l](#media:${encodeURIComponent(src)})`,
      `[l](${src})`,
      `${src}.bak`
    ]

    const viaRegex = cases(short).map(text => stripGeneratedImageEchoes(text, [short]))
    const viaScan = cases(long).map(text => stripGeneratedImageEchoes(text, [long]).split(long).join(short))

    expect(viaScan).toEqual(viaRegex)
  })
})
