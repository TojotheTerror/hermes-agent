import type { MediaImageDimensions } from '@/lib/media'

type ToolLike = {
  result?: unknown
  toolName?: unknown
  type?: unknown
}

type TextLike = {
  text?: unknown
  type?: unknown
}

// Path-ish result fields the model may echo into its prose. Display prefers the
// host path (gateway-deliverable); stripping must catch every variant so a
// sandbox path the model restated doesn't slip through as a duplicate image.
const DISPLAY_KEYS = ['host_image', 'image'] as const
const ECHO_KEYS = ['host_image', 'image', 'agent_visible_image'] as const

function recordFromUnknown(value: unknown): Record<string, unknown> | null {
  if (value && typeof value === 'object' && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }

  if (typeof value !== 'string' || !value.trim()) {
    return null
  }

  try {
    const parsed = JSON.parse(value)

    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as Record<string, unknown>) : null
  } catch {
    return null
  }
}

function stringFields(record: Record<string, unknown>, keys: readonly string[]): string[] {
  return keys.map(key => record[key]).filter((v): v is string => typeof v === 'string' && v.trim().length > 0)
}

function regexEscape(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

// V8 refuses a RegExp whose pattern is too large ("Regular expression too large"), and an
// image_generate result can carry an inline `data:` URL hundreds of KB long. A source longer than
// this is matched with plain string scans instead of a RegExp built around it; the scans keep the
// same boundaries as the patterns in stripGeneratedImageEchoes. Real paths and URLs stay far below.
const MAX_REGEX_SOURCE_LENGTH = 2048

// The parts of each echo pattern that don't contain the source, anchored to the source's position.
const IMAGE_BEFORE = /!\[[^\]\n]*\]\(\s*<?$/
const IMAGE_AFTER = /^>?(?=$|[\s)\\])[^)\n]*\)/
const MEDIA_BEFORE = /\[[^\]\n]*\]\(\s*#media:$/i
const MEDIA_AFTER = /^\s*\)/
const BARE_BOUNDARY = /[\s[{]/
const BARE_AFTER = /^>?(?=$|[\s\]}.,!?])/

// ASCII-only lowercase keeps every index in place (String#toLowerCase may change the length).
function asciiLower(value: string): string {
  return value.replace(/[A-Z]+/g, run => run.toLowerCase())
}

/** Remove every span `match` reports around an occurrence of `needle` (found with indexOf, ASCII
 *  case-insensitively when `ignoreCase`), continuing the scan from where each removal happened. */
function removeAround(
  text: string,
  needle: string,
  match: (text: string, at: number) => { end: number; start: number } | null,
  ignoreCase = false
): string {
  let next = text
  let from = 0
  const target = ignoreCase ? asciiLower(needle) : needle

  for (;;) {
    const at = (ignoreCase ? asciiLower(next) : next).indexOf(target, from)

    if (at < 0) {
      return next
    }

    const span = match(next, at)

    if (span) {
      next = next.slice(0, span.start) + next.slice(span.end)
      from = span.start
    } else {
      from = at + 1
    }
  }
}

function stripLongSourceEchoes(text: string, source: string): string {
  const encoded = encodeURIComponent(source)

  let next = removeAround(text, source, (value, at) => {
    const before = IMAGE_BEFORE.exec(value.slice(0, at))
    const after = before ? IMAGE_AFTER.exec(value.slice(at + source.length)) : null

    return before && after ? { start: before.index, end: at + source.length + after[0].length } : null
  })

  next = removeAround(
    next,
    encoded,
    (value, at) => {
      const before = MEDIA_BEFORE.exec(value.slice(0, at))
      const after = before ? MEDIA_AFTER.exec(value.slice(at + encoded.length)) : null

      return before && after ? { start: before.index, end: at + encoded.length + after[0].length } : null
    },
    true
  )

  return removeAround(next, source, (value, at) => {
    const start = at > 0 && value[at - 1] === '<' ? at - 1 : at
    const after = BARE_AFTER.exec(value.slice(at + source.length))

    return (start === 0 || BARE_BOUNDARY.test(value[start - 1])) && after
      ? { start, end: at + source.length + after[0].length }
      : null
  })
}

function unique(values: string[]): string[] {
  return [...new Set(values.filter(Boolean))]
}

function imageResult(part: ToolLike): Record<string, unknown> | null {
  if (part.type !== 'tool-call' || part.toolName !== 'image_generate') {
    return null
  }

  const record = recordFromUnknown(part.result)

  return record && record.success !== false ? record : null
}

/** Display source for a completed `image_generate` result (host path wins). */
export function generatedImageFromResult(result: unknown): string | null {
  const record = recordFromUnknown(result)

  if (!record || record.success === false) {
    return null
  }

  return stringFields(record, DISPLAY_KEYS)[0] ?? null
}

/** Image backends can report decoded PNG geometry (`pixel_size`) separately from the requested `size`.
 * Never use requested_size/size or aspect_ratio as intrinsic metadata. */
export function generatedImageDimensionsFromResult(result: unknown): MediaImageDimensions | undefined {
  const record = recordFromUnknown(result)
  const match = typeof record?.pixel_size === 'string' ? /^(\d+)x(\d+)$/.exec(record.pixel_size) : null

  return match &&
    Number.isSafeInteger(Number(match[1])) &&
    Number.isSafeInteger(Number(match[2])) &&
    Number(match[1]) > 0 &&
    Number(match[2]) > 0
    ? { width: Number(match[1]), height: Number(match[2]) }
    : undefined
}

/** Every path/URL a generated image might appear as in prose, for de-duping. */
export function generatedImageEchoSources(parts: readonly ToolLike[]): string[] {
  // Runs on every streamed tool event over the whole timeline; skip the
  // per-part array allocations for the (usual) rows that are not image results.
  const sources: string[] = []

  for (const part of parts) {
    const result = imageResult(part)

    if (result) {
      sources.push(...stringFields(result, ECHO_KEYS))
    }
  }

  return unique(sources)
}

/** Strip a generated image out of prose so it only ever shows in the tool slot.
 *  Once a generation succeeded (`sources` is non-empty) we drop embedded images
 *  and media links whose src/path matches one of the known generated sources —
 *  the model frequently restates the image in prose. Bare occurrences of the
 *  known paths/URLs are removed too. Surrounding prose is preserved, and
 *  references to files OTHER than the generated source (e.g. a separate
 *  `MEDIA:` delivery directive) are never touched (#80621). */
export function stripGeneratedImageEchoes(text: string, sources: readonly string[]): string {
  if (!text || sources.length === 0) {
    return text
  }

  let next = text

  for (const source of unique([...sources])) {
    if (source.length > MAX_REGEX_SOURCE_LENGTH) {
      next = stripLongSourceEchoes(next, source)

      continue
    }

    const escaped = regexEscape(source)
    // Markdown image whose src is this generated source: ![alt](<src> [title]).
    // The lookahead after the escaped source stops a prefix over-match — a
    // link to `/tmp/a.png.bak` must not be eaten by source `/tmp/a.png`.
    next = next.replace(new RegExp(String.raw`!\[[^\]\n]*\]\(\s*<?${escaped}>?(?=$|[\s)\\])[^)\n]*\)`, 'g'), '')
    // Media link whose path is this generated source: [label](#media:<encoded>).
    // Anchored to the closing paren so a link to a path that merely STARTS with
    // the encoded source (e.g. `#media:%2Ftmp%2Fa.png.bak` vs `/tmp/a.png`)
    // survives; case-insensitive so lowercase hex (`%2f`) is caught too. Only
    // whitespace is allowed before the closing paren — a raw space can never be
    // part of the href (encodeURIComponent turns it into %20), so this cannot
    // over-match a longer path.
    next = next.replace(
      new RegExp(String.raw`\[[^\]\n]*\]\(\s*#media:${regexEscape(encodeURIComponent(source))}\s*\)`, 'gi'),
      ''
    )
    // Bare occurrence of the source path/URL. `(`/`)` are excluded from the
    // boundary classes so a path inside a markdown link destination
    // (`[label](/tmp/a.png)`) is not corrupted into `[label]()`.
    next = next.replace(new RegExp(String.raw`(^|[\s[{])<?${escaped}>?(?=$|[\s\]}.,!?])`, 'g'), '$1')
  }

  return next
}

/** Strip generated-image echoes from text parts, dropping any part left empty.
 *  The image lives in the tool slot; prose keeps the agent's actual words. */
export function dedupeGeneratedImageEchoesInParts<T extends TextLike & ToolLike>(parts: T[]): T[] {
  const sources = generatedImageEchoSources(parts)

  // No echoes to strip: hand back the same array so React and the streaming
  // reducer see a no-op instead of a fresh identity on every tool event.
  if (!sources.length) {
    return parts
  }

  return parts
    .map(part =>
      part.type === 'text' && typeof part.text === 'string'
        ? { ...part, text: stripGeneratedImageEchoes(part.text, sources) }
        : part
    )
    .filter(part => part.type !== 'text' || (typeof part.text === 'string' && part.text.trim().length > 0))
}
