/**
 * Noesis — Ollama embedding client with auto-config and auto-pull
 */
export declare const FALLBACK_MODEL = "mxbai-embed-large";
export declare const DEFAULT_MODEL = "nomic-embed-text";
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
 * Auto-configure Ollama: verify reachability, check for model, pull if missing.
 * Falls back to FALLBACK_MODEL if primary model fails to pull.
 */
export declare function autoConfigOllama(endpoint: string, preferredModel: string, logger?: (msg: string) => void): Promise<OllamaClient>;
/**
 * Create an Ollama client (does not validate connectivity).
 */
export declare function createOllamaClient(endpoint: string, model: string): OllamaClient;
/**
 * Compute SHA-256 checksum of content + agentId (for deduplication).
 */
export declare function contentChecksum(content: string, agentId: string): string;
/**
 * Split text into overlapping chunks.
 */
export declare function chunkText(text: string, chunkSize: number, overlap: number): string[];
//# sourceMappingURL=ollama.d.ts.map