/**
 * Noesis — Local-first semantic memory plugin for OpenClaw
 *
 * Memory slot: plugins.slots.memory = "noesis"
 * Stack: LanceDB (vector store) + Ollama (embeddings) + BM25 + MMR
 * No cloud. No API keys. Fully local.
 */
declare const _default: {
    id: string;
    name: string;
    description: string;
    configSchema: import("openclaw/plugin-sdk/plugin-entry").OpenClawPluginConfigSchema;
    register: NonNullable<import("openclaw/plugin-sdk/plugin-entry").OpenClawPluginDefinition["register"]>;
} & Pick<import("openclaw/plugin-sdk/plugin-entry").OpenClawPluginDefinition, "kind" | "reload" | "nodeHostCommands" | "securityAuditCollectors">;
export default _default;
//# sourceMappingURL=index.d.ts.map