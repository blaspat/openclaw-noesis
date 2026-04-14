/**
 * Noesis — Type definitions
 */
export const DEFAULT_CONFIG = {
    lanceDbPath: "~/.openclaw/noesis/db",
    ollamaEndpoint: "http://localhost:11434",
    embeddingModel: "nomic-embed-text",
    chunkSize: 512,
    chunkOverlap: 64,
    topK: 6,
    autoMigrate: false,
    indexQmdSessions: true,
    watchMemoryDirs: false,
    gitLfsEnabled: false,
    gitLfsRepo: "blaspat/openclaw-noesis-data",
    annNprobe: 16,
    annNumSubvectors: 96,
    defaultTtlDays: 90,
    autoCleanup: true,
    cleanupIntervalHours: 6,
    contextEngineEnabled: true,
    assembleInjectPriority: 75,
    assembleMaxEntries: 20,
    assembleMaxAgeDays: 30,
};
//# sourceMappingURL=types.js.map