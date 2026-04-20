/**
 * Noesis — HuggingFace Transformers.js embedding client
 * Replaces Ollama with in-process WASM/SIMD embeddings for CPU-only deployments.
 *
 * Auto-downloads the model from HuggingFace on first use (no Ollama needed).
 * Graceful degradation: returns zero vectors on embed failure.
 */

import { createHash } from "crypto";

// Lazy-load the transformers package to avoid blocking startup if not installed
let _transformers: typeof import("@huggingface/transformers") | null = null;

async function getTransformers(): Promise<typeof import("@huggingface/transformers")> {
  if (_transformers) return _transformers;
  try {
    _transformers = await import("@huggingface/transformers");
    return _transformers;
  } catch {
    throw new Error(
      "[noesis/transformers] @huggingface/transformers not found. " +
      "Run: npm install @huggingface/transformers"
    );
  }
}

export const DEFAULT_HF_MODEL = "nomic-ai/nomic-embed-text-v1.5";
export const HF_MODEL_VARIANT = "onnx/model_q4f16.onnx";

export interface EmbeddingClient {
  endpoint: string;   // kept for interface compat — value is the model ID
  model: string;      // kept for interface compat — value is the model ID
  embed(text: string, prefix?: string): Promise<number[]>;
  embedBatch(texts: string[], prefix?: string): Promise<number[][]>;
}

/**
 * Auto-configure the Transformers.js embedding client.
 * Downloads the model from HuggingFace on first run (cached locally).
 */
export async function autoConfigTransformers(
  modelId: string,
  logger?: (msg: string) => void,
  signal?: AbortSignal
): Promise<EmbeddingClient> {
  const transformers = await getTransformers();

  // Build the full model ID with ONNX variant
  const fullModelId = `${modelId.replace(/\/$/, "")}/${HF_MODEL_VARIANT}`;
  logger?.(`[noesis/transformers] Loading embedding model: ${fullModelId}`);

  // Configure env for Node.js — disable browser cache, allow local files
  const { env } = transformers;
  env.allowLocalModels = false;
  env.useBrowserCache = false;

  let pipeline: Awaited<ReturnType<typeof transformers.pipeline>> | null = null;
  let pipelineError: Error | null = null;

  // Load pipeline with timeout
  try {
    const loadPromise = transformers.pipeline(
      "feature-extraction",
      fullModelId,
      {
        device: "cpu",
      } as any
    );

    // Race the load against a timeout
    pipeline = await Promise.race([
      loadPromise,
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("Model load timeout (60s)")), 60_000)
      ),
    ]) as Awaited<ReturnType<typeof transformers.pipeline>>;

    logger?.(`[noesis/transformers] Model ready: ${fullModelId}`);
  } catch (err) {
    pipelineError = err as Error;
    logger?.(`[noesis/transformers] Failed to load model: ${pipelineError.message}`);
    // Continue — we still return a client that returns zero vectors
  }

  return createTransformersClient(pipeline, modelId, fullModelId, pipelineError, logger);
}

function createTransformersClient(
  pipeline: Awaited<ReturnType<typeof import("@huggingface/transformers").pipeline>> | null,
  _modelId: string,
  _fullModelId: string,
  _loadError: Error | null,
  logger?: (msg: string) => void
): EmbeddingClient {
  // Default embedding dimension: 768 (nomic-embed-text v1.5)
  const EMBED_DIM = 768;

  async function embed(text: string, prefix = ""): Promise<number[]> {
    if (!pipeline) {
      logger?.(`[noesis/transformers] embed() called but pipeline not loaded — returning zero vector`);
      return new Array(EMBED_DIM).fill(0);
    }

    const prefixed = prefix ? `${prefix} ${text}` : text;

    // Race inference against a timeout
    try {
      const result = await Promise.race([
        pipeline(prefixed, { pooling: "mean", normalize: true } as any) as Promise<any>,
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("Embed timeout (30s)")), 30_000)
        ),
      ]);

      // Extract embedding array from the Tensor-like result
      const data = result?.data ?? result?.tolist?.() ?? result?.toList?.();
      if (!data || !Array.isArray(data)) {
        logger?.(`[noesis/transformers] embed() returned unexpected format — returning zero vector`);
        return new Array(EMBED_DIM).fill(0);
      }
      return Array.isArray(data) ? data : new Array(EMBED_DIM).fill(0);
    } catch (err) {
      logger?.(`[noesis/transformers] embed() failed: ${(err as Error).message} — returning zero vector`);
      return new Array(EMBED_DIM).fill(0);
    }
  }

  async function embedBatch(texts: string[], prefix = ""): Promise<number[][]> {
    if (!pipeline) {
      logger?.(`[noesis/transformers] embedBatch() called but pipeline not loaded — returning zero vectors`);
      return texts.map(() => new Array(EMBED_DIM).fill(0));
    }

    // Run embeddings sequentially with concurrency control
    // Transformers.js doesn't have native batch, so parallelize
    const CONCURRENCY = 4;
    const results: number[][] = new Array(texts.length);

    for (let i = 0; i < texts.length; i += CONCURRENCY) {
      const batch = texts.slice(i, i + CONCURRENCY);
      const settled = await Promise.allSettled(batch.map((t) => embed(t, prefix)));

      for (let j = 0; j < settled.length; j++) {
        const item = settled[j] as PromiseFulfilledResult<number[]> | PromiseRejectedResult;
        if (item.status === "fulfilled") {
          results[i + j] = item.value;
        } else {
          logger?.(`[noesis/transformers] embedBatch item ${i + j} failed: ${(item as PromiseRejectedResult).reason?.message} — zero vector`);
          results[i + j] = new Array(EMBED_DIM).fill(0);
        }
      }
    }

    return results;
  }

  return {
    get endpoint() { return _fullModelId; },
    get model() { return _modelId; },
    embed,
    embedBatch,
  };
}

// ─── Shared utilities (formerly in ollama.ts) ────────────────────────────────

/**
 * Compute SHA-256 checksum of content + agentId (for deduplication).
 */
export function contentChecksum(content: string, agentId: string): string {
  return createHash("sha256").update(content).update(agentId).digest("hex");
}

// ─── Smart markdown chunking ────────────────────────────────────────────────

/**
 * Break-point patterns for smart chunking.
 * Higher scores = better place to cut.
 * Order matters — more specific patterns come first.
 */
const BREAK_PATTERNS: Array<[RegExp, number, string]> = [
  [/\n```/g,              80, "codeblock"],  // code fence boundary
  [/\n#{1}(?!#)[^\n]*\n/g,  70, "h1"],      // top-level heading (not ##)
  [/\n#{2}(?!#)[^\n]*\n/g,  60, "h2"],      // h2
  [/\n#{3,6}[^\n]*\n/g,   50, "h3+"],     // h3-h6
  [/\n(?:---|-{3,}|\*{3,}|_{3,})\s*\n/g, 40, "hr"],  // horizontal rule
  [/\n\n+/g,              20, "blank"],    // paragraph boundary
  [/\n[-*+][ \t]/g,       8, "ul"],      // unordered list
  [/\n\d+[.)][ \t]/g,      8, "ol"],      // ordered list
];

/**
 * Approximate characters per word (used to convert word-based config to char ranges).
 */
const CHARS_PER_WORD = 5;


interface BreakPoint {
  pos: number;
  score: number;
  type: string;
}

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

function findCodeFences(text: string): Array<{ start: number; end: number }> {
  const fences: Array<{ start: number; end: number }> = [];
  const pattern = /\n```/g;
  let lastPos = 0;
  for (const match of text.matchAll(pattern)) {
    const pos = match.index!;
    if (lastPos === 0) {
      lastPos = pos;
    } else {
      fences.push({ start: lastPos, end: pos + 4 });
      lastPos = 0;
    }
  }
  return fences;
}

function inCodeFence(pos: number, fences: Array<{ start: number; end: number }>): boolean {
  for (const fence of fences) {
    if (pos >= fence.start && pos < fence.end) return true;
  }
  return false;
}

function findBestCutoff(
  breakPoints: BreakPoint[],
  targetEnd: number,
  windowChars: number,
  codeFences: Array<{ start: number; end: number }>,
  wordPositions: number[]
): number {
  const windowStart = Math.max(0, targetEnd - windowChars);
  let best = -1;
  let bestScore = 0;
  for (const bp of breakPoints) {
    let pos = bp.pos;
    if (pos < windowStart || pos > targetEnd) continue;
    if (pos <= windowStart + 20) continue;
    if (inCodeFence(pos, codeFences)) continue;
    for (const fence of codeFences) {
      if (pos > fence.start && pos < fence.end) { pos = -1; break; }
    }
    if (pos < 0) continue;
    const distance = targetEnd - pos;
    const distanceRatio = distance / windowChars;
    const score = bp.score * (1 - distanceRatio * distanceRatio * 0.7);
    if (score > bestScore) { bestScore = score; best = pos; }
  }
  return best;
}

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
 * Respects headings, code fences, paragraph breaks, and list items.
 */
export function chunkText(text: string, chunkSize: number, overlap: number): string[] {
  const trimmed = text.trim();
  if (!trimmed) return [];
  const words = trimmed.split(/\s+/);
  if (words.length <= chunkSize) return [trimmed];

  const breakPoints = scanBreakPoints(trimmed);
  const codeFences = findCodeFences(trimmed);
  const maxChars = Math.floor(chunkSize * CHARS_PER_WORD * 1.5);
  const windowChars = Math.floor(chunkSize * CHARS_PER_WORD * 0.5);
  const minChars = Math.floor(chunkSize * CHARS_PER_WORD * 0.3);

  const wordChars: number[] = [0];
  let offset = 0;
  for (let i = 0; i < words.length; i++) {
    offset += words[i].length;
    wordChars.push(offset);
    if (i < words.length - 1) offset += 1;
  }

  const chunks: string[] = [];
  let charPos = 0;
  while (charPos < trimmed.length) {
    const targetEndPos = charPos + maxChars;
    let endPos = Math.min(targetEndPos, trimmed.length);
    if (endPos < trimmed.length) {
      const best = findBestCutoff(breakPoints, targetEndPos, windowChars, codeFences, wordChars);
      if (best > charPos + minChars) endPos = best;
    }
    if (endPos <= charPos) {
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
    const endWordIdx = wordIndexAtPos(endPos, wordChars);
    const startWordIdx = wordIndexAtPos(charPos, wordChars);
    charPos = wordChars[Math.min(startWordIdx + Math.max(0, endWordIdx - startWordIdx - overlap), wordChars.length - 1)];
    if (charPos >= endPos) charPos = endPos;
  }
  return chunks.filter(Boolean);
}

