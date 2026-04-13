# openclaw-noesis

Semantic memory plugin for OpenClaw — cross-session recall, embedding-based search, and memory indexing per agent session.

## Features

- **Semantic search** — Stores memory entries with embeddings and finds similar entries using cosine similarity
- **Cross-session recall** — Query memories from any past session or agent without prompts
- **Per-agent indexing** — Memories are indexed by agent ID for targeted recall
- **In-memory vector store** — Fast, zero-dependency vector operations (swap for LanceDB or pgvector in production)
- **Plugin tools** — Four agent tools: `noesis_index`, `noesis_search`, `noesis_recall`, `noesis_status`

## Installation

```bash
npm install openclaw-noesis
```

Or link locally during development:

```bash
cd openclaw-noesis
npm install
npm run build
openclaw plugins install ./openclaw-noesis
```

## Configuration

Add to your OpenClaw config:

```json
{
  "plugins": {
    "entries": {
      "noesis": {
        "enabled": true,
        "config": {
          "embeddingProvider": "openai",
          "embeddingModel": "text-embedding-3-small",
          "topK": 5,
          "indexOnLoad": true
        }
      }
    }
  }
}
```

| Option | Default | Description |
|--------|---------|-------------|
| `embeddingProvider` | `openai` | Embedding provider to use (openai, gemini, etc.) |
| `embeddingModel` | `text-embedding-3-small` | Embedding model name |
| `topK` | `5` | Default number of search results to return |
| `indexOnLoad` | `true` | Whether to re-index existing memories on plugin load |

## Tools

### `noesis_index`

Store a new memory entry. Embeds and indexes the content automatically.

```
noesis_index({ content: "Patrick prefers short messages", agentId: "claire" })
```

### `noesis_search`

Semantic search across all stored memories.

```
noesis_search({ query: "Patrick's communication preferences", topK: 3 })
```

### `noesis_recall`

Recall memories from a specific session or agent.

```
noesis_recall({ agentId: "claire", limit: 10 })
```

### `noesis_status`

Get current plugin status and total entry count.

```
noesis_status({})
```

## Architecture

The plugin uses a simple in-memory vector store. Each entry gets a 1536-dimension embedding generated deterministically from the content. Search uses cosine similarity to rank candidates.

In production, replace `generateEmbedding()` with a real API call to your embedding provider (OpenAI, Gemini, etc.) and swap `VectorStore` for a proper vector DB like LanceDB or pgvector.

## Development

```bash
npm install
npm run build    # compile TypeScript → dist/
npm run dev      # watch mode
```

## License

MIT
