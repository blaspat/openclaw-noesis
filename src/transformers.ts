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

// ─── Re-export shared utilities from ollama.ts for compat ────────────────────

export { contentChecksum, chunkText } from "./ollama.js";
