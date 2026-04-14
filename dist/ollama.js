/**
 * Noesis — Ollama embedding client with auto-config and auto-pull
 */
import { createHash } from "crypto";
export const FALLBACK_MODEL = "mxbai-embed-large";
export const DEFAULT_MODEL = "nomic-embed-text";
/**
 * Check if Ollama is reachable and return available model list.
 */
async function listModels(endpoint) {
    const res = await fetch(`${endpoint}/api/tags`, {
        signal: AbortSignal.timeout(5000),
    });
    if (!res.ok)
        throw new Error(`Ollama /api/tags returned ${res.status}`);
    const data = (await res.json());
    return (data.models ?? []).map((m) => ({ name: m.name, size: m.size }));
}
/**
 * Pull a model from Ollama (streaming response, waits for completion).
 */
async function pullModel(endpoint, model, logger) {
    logger?.(`[noesis/ollama] Pulling model ${model} — this may take a few minutes...`);
    const res = await fetch(`${endpoint}/api/pull`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: model, stream: true }),
        signal: AbortSignal.timeout(600_000), // 10-minute timeout for large models
    });
    if (!res.ok)
        throw new Error(`Ollama /api/pull returned ${res.status}`);
    // Consume the NDJSON stream to wait for completion
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let lastStatus = "";
    while (true) {
        const { done, value } = await reader.read();
        if (done)
            break;
        const chunk = decoder.decode(value, { stream: true });
        for (const line of chunk.split("\n")) {
            if (!line.trim())
                continue;
            try {
                const obj = JSON.parse(line);
                if (obj.error)
                    throw new Error(`Pull error: ${obj.error}`);
                if (obj.status && obj.status !== lastStatus) {
                    lastStatus = obj.status;
                    logger?.(`[noesis/ollama] ${model}: ${obj.status}`);
                }
            }
            catch {
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
export async function autoConfigOllama(endpoint, preferredModel, logger) {
    // 1. Check connectivity
    let models;
    try {
        models = await listModels(endpoint);
        logger?.(`[noesis/ollama] Connected to Ollama at ${endpoint} (${models.length} model(s) available)`);
    }
    catch (err) {
        throw new Error(`[noesis/ollama] Cannot connect to Ollama at ${endpoint}. Is 'ollama serve' running? Error: ${err}`);
    }
    // 2. Check if preferred model is available, pull if not
    const modelNames = new Set(models.map((m) => m.name.split(":")[0]));
    const preferredBase = preferredModel.split(":")[0];
    let resolvedModel = preferredModel;
    if (!modelNames.has(preferredBase)) {
        logger?.(`[noesis/ollama] Model '${preferredModel}' not found. Pulling...`);
        try {
            await pullModel(endpoint, preferredModel, logger);
        }
        catch (pullErr) {
            logger?.(`[noesis/ollama] Failed to pull '${preferredModel}': ${pullErr}. Trying fallback: ${FALLBACK_MODEL}`);
            try {
                await pullModel(endpoint, FALLBACK_MODEL, logger);
                resolvedModel = FALLBACK_MODEL;
            }
            catch (fallbackErr) {
                throw new Error(`[noesis/ollama] Failed to pull both '${preferredModel}' and '${FALLBACK_MODEL}'. ` +
                    `Please run 'ollama pull ${preferredModel}' manually. Error: ${fallbackErr}`);
            }
        }
    }
    logger?.(`[noesis/ollama] Using embedding model: ${resolvedModel}`);
    return createOllamaClient(endpoint, resolvedModel);
}
/**
 * Create an Ollama client (does not validate connectivity).
 */
export function createOllamaClient(endpoint, model) {
    async function embed(text) {
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
        const data = (await res.json());
        if (!data.embedding)
            throw new Error("Ollama returned no embedding vector");
        return data.embedding;
    }
    async function embedBatch(texts) {
        // Ollama doesn't have native batch; parallelize with concurrency cap
        const CONCURRENCY = 4;
        const results = new Array(texts.length);
        for (let i = 0; i < texts.length; i += CONCURRENCY) {
            const batch = texts.slice(i, i + CONCURRENCY);
            const embeddings = await Promise.all(batch.map((t) => embed(t)));
            for (let j = 0; j < embeddings.length; j++) {
                results[i + j] = embeddings[j];
            }
        }
        return results;
    }
    return { endpoint, model, embed, embedBatch };
}
/**
 * Compute SHA-256 checksum of content + agentId (for deduplication).
 */
export function contentChecksum(content, agentId) {
    return createHash("sha256").update(content).update(agentId).digest("hex");
}
/**
 * Split text into overlapping chunks.
 */
export function chunkText(text, chunkSize, overlap) {
    const words = text.split(/\s+/);
    if (words.length <= chunkSize)
        return [text.trim()].filter(Boolean);
    const chunks = [];
    let start = 0;
    while (start < words.length) {
        const end = Math.min(start + chunkSize, words.length);
        chunks.push(words.slice(start, end).join(" "));
        if (end === words.length)
            break;
        start += chunkSize - overlap;
    }
    return chunks.filter(Boolean);
}
//# sourceMappingURL=ollama.js.map