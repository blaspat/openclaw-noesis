/**
 * Noesis — Ollama embedding client with auto-config and auto-pull
 */

import { createHash } from "crypto";

export const FALLBACK_MODEL = "mxbai-embed-large";
export const DEFAULT_MODEL = "nomic-embed-text";

export interface OllamaModel {
  name: string;
  size: number;
}

export interface OllamaClient {
  endpoint: string;
  model: string;
  embed(text: string): Promise<number[]>;
  embedBatch(texts: string[]): Promise<number[][]>;
}

/**
 * Check if Ollama is reachable and return available model list.
 */
async function listModels(endpoint: string): Promise<OllamaModel[]> {
  const res = await fetch(`${endpoint}/api/tags`, {
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`Ollama /api/tags returned ${res.status}`);
  const data = (await res.json()) as { models?: Array<{ name: string; size: number }> };
  return (data.models ?? []).map((m) => ({ name: m.name, size: m.size }));
}

/**
 * Pull a model from Ollama (streaming response, waits for completion).
 */
async function pullModel(endpoint: string, model: string, logger?: (msg: string) => void): Promise<void> {
  logger?.(`[noesis/ollama] Pulling model ${model} — this may take a few minutes...`);
  const res = await fetch(`${endpoint}/api/pull`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: model, stream: true }),
    signal: AbortSignal.timeout(600_000), // 10-minute timeout for large models
  });
  if (!res.ok) throw new Error(`Ollama /api/pull returned ${res.status}`);

  // Consume the NDJSON stream to wait for completion
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let lastStatus = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    for (const line of chunk.split("\n")) {
      if (!line.trim()) continue;
      try {
        const obj = JSON.parse(line) as { status?: string; error?: string };
        if (obj.error) throw new Error(`Pull error: ${obj.error}`);
        if (obj.status && obj.status !== lastStatus) {
          lastStatus = obj.status;
          logger?.(`[noesis/ollama] ${model}: ${obj.status}`);
        }
      } catch {
        // ignore JSON parse errors in stream
      }
    }
  }
  logger?.(`[noesis/ollama] Model ${model} ready.`);
}

/**
 * Auto-configure Ollama: verify reachability, check for model, pull if missing.
 * Falls back to FALLBACK_MODEL if primary model fails to pull.
 */
export async function autoConfigOllama(
  endpoint: string,
  preferredModel: string,
  logger?: (msg: string) => void
): Promise<OllamaClient> {
  // 1. Check connectivity
  let models: OllamaModel[];
  try {
    models = await listModels(endpoint);
    logger?.(`[noesis/ollama] Connected to Ollama at ${endpoint} (${models.length} model(s) available)`);
  } catch (err) {
    throw new Error(
      `[noesis/ollama] Cannot connect to Ollama at ${endpoint}. Is 'ollama serve' running? Error: ${err}`
    );
  }

  // 2. Check if preferred model is available, pull if not
  const modelNames = new Set(models.map((m) => m.name.split(":")[0]));
  const preferredBase = preferredModel.split(":")[0];

  let resolvedModel = preferredModel;

  if (!modelNames.has(preferredBase)) {
    logger?.(`[noesis/ollama] Model '${preferredModel}' not found. Pulling...`);
    try {
      await pullModel(endpoint, preferredModel, logger);
    } catch (pullErr) {
      logger?.(`[noesis/ollama] Failed to pull '${preferredModel}': ${pullErr}. Trying fallback: ${FALLBACK_MODEL}`);
      try {
        await pullModel(endpoint, FALLBACK_MODEL, logger);
        resolvedModel = FALLBACK_MODEL;
      } catch (fallbackErr) {
        throw new Error(
          `[noesis/ollama] Failed to pull both '${preferredModel}' and '${FALLBACK_MODEL}'. ` +
            `Please run 'ollama pull ${preferredModel}' manually. Error: ${fallbackErr}`
        );
      }
    }
  }

  logger?.(`[noesis/ollama] Using embedding model: ${resolvedModel}`);
  return createOllamaClient(endpoint, resolvedModel);
}

/**
 * Create an Ollama client (does not validate connectivity).
 */
export function createOllamaClient(endpoint: string, model: string): OllamaClient {
  async function embed(text: string): Promise<number[]> {
    try {
      const res = await fetch(`${endpoint}/api/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, prompt: text }),
        signal: AbortSignal.timeout(30_000),
      });
      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Ollama /api/embeddings error ${res.status}: ${body}`);
      }
      const data = (await res.json()) as { embedding?: number[] };
      if (!data.embedding) throw new Error("Ollama returned no embedding vector");
      return data.embedding;
    } catch (err) {
      // Graceful degradation: return zero vector on embed failure.
      // Caller can detect via all-zero embedding.
      const dim = model.includes("mxbai") ? 1024 : 768;
      return new Array(dim).fill(0);
    }
  }

  async function embedBatch(texts: string[]): Promise<number[][]> {
    // Ollama doesn't have native batch; parallelize with concurrency cap.
    // Individual failures are tolerated — failed items get a zero-vector placeholder
    // so the batch can continue rather than propagate the error upward.
    const CONCURRENCY = 4;
    const results: number[][] = new Array(texts.length);
    for (let i = 0; i < texts.length; i += CONCURRENCY) {
      const batch = texts.slice(i, i + CONCURRENCY);
      const batchResults = await Promise.allSettled(batch.map((t) => embed(t)));
      for (let j = 0; j < batchResults.length; j++) {
        const result = batchResults[j];
        if (result.status === "fulfilled") {
          results[i + j] = result.value;
        } else {
          // Embed failed — use zero vector as placeholder; caller can detect
          const dim = model.includes("mxbai") ? 1024 : 768;
          results[i + j] = new Array(dim).fill(0);
        }
      }
    }
    return results;
  }

  return { endpoint, model, embed, embedBatch };
}

/**
 * Compute SHA-256 checksum of content + agentId (for deduplication).
 */
export function contentChecksum(content: string, agentId: string): string {
  return createHash("sha256").update(content).update(agentId).digest("hex");
}

// ─── smart markdown chunking ────────────────────────────────────────────────

/**
 * Break-point patterns for smart chunking.
 * Higher scores = better place to cut.
 * Order matters — more specific patterns come first.
 */
const BREAK_PATTERNS: Array<[RegExp, number, string]> = [
  [/\n```/g,           80, "codeblock"],   // code fence boundary
  [/\n#{1}(?!#)[^\n]*\n/g,  70, "h1"],     // top-level heading (not ##)
  [/\n#{2}(?!#)[^\n]*\n/g,  60, "h2"],     // h2
  [/\n#{3,6}[^\n]*\n/g,   50, "h3+"],    // h3-h6
  [/\n(?:---|-{3,}|\*{3,}|_{3,})\s*\n/g, 40, "hr"],  // horizontal rule
  [/\n\n+/g,           20, "blank"],     // paragraph boundary
  [/\n[-*+][ \t]/g,   8, "ul"],       // unordered list
  [/\n\d+[.)][ \t]/g,  8, "ol"],       // ordered list
];

/**
 * Approximate characters per word (used to convert word-based config to char ranges).
 * This is conservative — actual text varies, so we use a conservative multiplier.
 */
const CHARS_PER_WORD = 5;

interface BreakPoint {
  pos: number;   // character offset in original text
  score: number;
  type: string;
}

/**
 * Find all break-point candidates in text, deduplicated by position.
 * When multiple patterns match the same position, the highest score wins.
 */
function scanBreakPoints(text: string): BreakPoint[] {
  const seen = new Map<number, BreakPoint>();

  for (const [pattern, score, type] of BREAK_PATTERNS) {
    for (const match of text.matchAll(pattern)) {
      const pos = match.index!;
      const existing = seen.get(pos);
      if (!existing || score > existing.score) {
        seen.set(pos, { pos, score, type });
      }
    }
  }

  return Array.from(seen.values()).sort((a, b) => a.pos - b.pos);
}

/**
 * Find all code-fence regions so we never split inside them.
 */
function findCodeFences(text: string): Array<{ start: number; end: number }> {
  const fences: Array<{ start: number; end: number }> = [];
  const pattern = /\n```/g;
  let lastPos = 0;

  for (const match of text.matchAll(pattern)) {
    const pos = match.index!;
    if (lastPos === 0) {
      lastPos = pos;
    } else {
      fences.push({ start: lastPos, end: pos + 4 }); // +4 for "```\n"
      lastPos = 0;
    }
  }
  return fences;
}

/**
 * Check if a position falls inside a code fence region.
 */
function inCodeFence(pos: number, fences: Array<{ start: number; end: number }>): boolean {
  for (const fence of fences) {
    if (pos >= fence.start && pos < fence.end) return true;
  }
  return false;
}

/**
 * Find the best break point within a window near targetEnd.
 * score = baseScore * (1 - (distance/window)² * 0.7).
 * Skips positions inside code fences and at/near window start.
 */
function findBestCutoff(
  breakPoints: BreakPoint[],
  targetEnd: number,
  windowSize: number,
  codeFences: Array<{ start: number; end: number }>,
  minWords: number,
  wordPositions: number[]
): number {
  const windowStart = Math.max(0, targetEnd - windowSize);

  let best = -1;
  let bestScore = 0;

  for (const bp of breakPoints) {
    const pos = bp.pos;

    // Must be in window and before or at targetEnd
    if (pos < windowStart || pos > targetEnd) continue;
    // Avoid cutting at the very start of the window (too close to last chunk boundary)
    if (pos <= windowStart + 20) continue;
    // Skip inside code fences
    if (inCodeFence(pos, codeFences)) continue;

    // Compute distance penalty
    const distance = targetEnd - pos;
    const distanceRatio = distance / windowSize;
    const score = bp.score * (1 - distanceRatio * distanceRatio * 0.7);

    // Also ensure we wouldn't split a code fence open — if the fence opens before
    // targetEnd but closes after windowStart, cutting at pos would split it
    let splitsOpenFence = false;
    for (const fence of codeFences) {
      if (pos > fence.start && pos < fence.end) {
        splitsOpenFence = true;
        break;
      }
    }
    if (splitsOpenFence) continue;

    if (score > bestScore) {
      bestScore = score;
      best = pos;
    }
  }

  return best;
}

/**
 * Get word index at a character position (binary search on precomputed word offsets).
 */
function wordIndexAtPos(charPos: number, wordPositions: number[]): number {
  let lo = 0, hi = wordPositions.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    if (wordPositions[mid] < charPos) lo = mid + 1;
    else hi = mid - 1;
  }
  return lo;
}

/**
 * Split text into overlapping chunks using smart markdown-aware boundary detection.
 *
 * Each chunk targets ~chunkSize words but respects natural markdown boundaries:
 * headings, code fences, paragraph breaks, and list items take precedence over
 * hard cuts in the middle of a semantic unit.
 *
 * Falls back to word-based splitting when no good break point exists within
 * the search window near the target boundary.
 *
 * @param text          - text to chunk
 * @param chunkSize     - target chunk size in words (soft upper bound)
 * @param overlap        - overlap in words between consecutive chunks
 */
export function chunkText(text: string, chunkSize: number, overlap: number): string[] {
  const trimmed = text.trim();
  if (!trimmed) return [];

  // Fast path: text fits in one chunk
  const words = trimmed.split(/\s+/);
  if (words.length <= chunkSize) return [trimmed];

  // Precompute
  const breakPoints = scanBreakPoints(trimmed);
  const codeFences = findCodeFences(trimmed);

  // Approximate char equivalents for window sizing
  // Use chunkSize * 1.5 chars as the soft upper bound per chunk
  const maxChars = Math.floor(chunkSize * CHARS_PER_WORD * 1.5);
  const windowChars = Math.floor(chunkSize * CHARS_PER_WORD * 0.5); // 50% of chunkSize in chars as window
  const minChars = Math.floor(chunkSize * CHARS_PER_WORD * 0.3);   // minimum 30% of chunkSize chars

  // Precompute character offset of each word boundary for word-index lookups
  // (approximate: iterate and count chars per word + trailing space)
  const wordChars: number[] = [0];
  let offset = 0;
  for (let i = 0; i < words.length; i++) {
    offset += words[i].length;
    wordChars.push(offset);
    if (i < words.length - 1) offset += 1; // trailing space
  }

  const chunks: string[] = [];
  let charPos = 0;

  while (charPos < trimmed.length) {
    const targetEndPos = charPos + maxChars;
    let endPos = Math.min(targetEndPos, trimmed.length);

    if (endPos < trimmed.length) {
      // Try to find a natural markdown break point near targetEndPos
      const bestCutoff = findBestCutoff(
        breakPoints,
        targetEndPos,
        windowChars,
        codeFences,
        chunkSize,
        wordChars
      );

      if (bestCutoff > charPos + minChars) {
        endPos = bestCutoff;
      }
    }

    // Fallback: if endPos is at or before charPos, force a word boundary
    if (endPos <= charPos) {
      // Scan forward to accumulate chunkSize words
      const wordIdx = wordIndexAtPos(charPos, wordChars);
      const endWordIdx = Math.min(wordIdx + chunkSize, words.length);
      const rawChunk = words.slice(wordIdx, endWordIdx).join(" ");
      endPos = charPos + rawChunk.length;
      chunks.push(rawChunk);
      if (endWordIdx >= words.length) break;
      charPos = wordChars[Math.max(0, endWordIdx - overlap)];
      continue;
    }

    const chunk = trimmed.slice(charPos, endPos).trim();
    if (chunk) chunks.push(chunk);

    if (endPos >= trimmed.length) break;

    // Overlap: back up by `overlap` words worth of characters
    const overlapWordIdx = wordIndexAtPos(charPos, wordChars) +
      Math.max(0, wordIndexAtPos(endPos, wordChars) - wordIndexAtPos(charPos, wordChars) - overlap);
    charPos = wordChars[Math.min(overlapWordIdx, wordChars.length - 1)];

    // Safety: if charPos hasn't moved forward (tiny overlap window), advance by minChars
    if (charPos <= endPos - minChars) {
      // normal case — charPos is before the safe threshold, we're good
    } else {
      // force forward progress
      charPos = wordChars[Math.min(wordIndexAtPos(endPos, wordChars) - overlap, wordChars.length - 1)];
      if (charPos <= charPos) { // still stuck
        charPos = endPos;
      }
    }
  }

  return chunks.filter(Boolean);
}
